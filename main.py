from components.LLM import rLLM
from components.Database import AdvancedClient
from components.utils import get_preview_pdf, create_refrences
import streamlit as st
import base64

TOGETHER_API = st.secrets["TOGETHER_API_KEY"]

st.set_page_config("wide")

# session state defined
if "llm_tokens" not in st.session_state:
    st.session_state.llm_tokens = {"Input": 0, "Output": 0}
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "client" not in st.session_state:
    st.session_state.client = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "landing_container" not in st.session_state:
    st.session_state.landing_container = st.empty()


def main():

    if st.session_state.client == None:
        st.session_state.client = AdvancedClient(vector_database_path="vectorDB")

    if not st.session_state.file_uploaded:
        with st.session_state.landing_container.container():
            with open("static/heading.html", "r") as file:
                upper_html = file.read()
            st.markdown(upper_html, unsafe_allow_html=True)
            uploaded_files = st.file_uploader("Upload PDF.", accept_multiple_files=True)

            if st.button("Next"):
                if uploaded_files:
                    file_names, file_types, file_datas = [], [], []
                    for file in uploaded_files:

                        file_name = file.name
                        file_type = file_name.split(".")[1]
                        file_data = file.getvalue()

                        file_names.append(file_name)
                        file_datas.append(file_data)
                        file_types.append(file_type)

                        file_to_preview = (
                            [file_name, get_preview_pdf(file_bytes=file_data)]
                            if file_type == "pdf"
                            else [file_name]
                        )
                        st.session_state.uploaded_files.append(file_to_preview)
                    # populate the vector database
                    st.session_state.client.create_or_get_collection(
                        file_names, file_types, file_datas
                    )
                    st.session_state.file_uploaded = True
                else:
                    st.session_state.file_uploaded = True

            with open("static/bottom.html", "r") as file:
                lower_html = file.read()
            st.markdown("\n\n" + lower_html, unsafe_allow_html=True)

    if st.session_state.file_uploaded:
        st.session_state.landing_container.empty()
        with open("static/heading.html", "r") as file:
            upper_html_ = file.read()
        st.markdown(
            upper_html_.replace("64", "55").replace("24", "20"), unsafe_allow_html=True
        )
        llm = rLLM(llm_name="meta-llama/Llama-3-8b-chat-hf", api_key=TOGETHER_API)
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.markdown(message["content"])
                    with st.expander("References"):
                        st.markdown(message["refrences"], unsafe_allow_html=True)
                else:
                    st.markdown(message["content"])

        query = st.chat_input()

        if query:
            st.session_state.chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.write(query)

            rephrased_query = llm.HyDE(query, st.session_state.chat_history)
            retrieved_docs = st.session_state.client.retrieve_chunks(
                query=rephrased_query
            )

            context = ""
            for i, doc in enumerate(retrieved_docs, start=1):
                context += f"Chunk {i}\n\n" + doc["document"] + "\n\n"

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                for data in llm.generate_rag_response(
                    context, query, st.session_state.chat_history
                ):
                    completed, chunk = data
                    if completed:
                        _, input_tokens, output_tokens = chunk
                    else:
                        full_response += chunk  # type: ignore
                        response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

                with st.expander(label="References"):
                    refrences = create_refrences(retrieved_docs)
                    st.markdown(refrences, unsafe_allow_html=True)

            st.session_state.llm_tokens["Output"] += output_tokens
            st.session_state.llm_tokens["Input"] += input_tokens
            st.session_state.chat_history.append(
                {"role": "assistant", "content": full_response, "refrences": refrences}
            )

    with st.sidebar:
        st.markdown("# Files in Use")

        if len(st.session_state.uploaded_files) > 0:
            for doc in st.session_state.uploaded_files:
                if len(doc) > 1:
                    with st.expander(label=doc[0]):
                        doc_bytes = doc[1]
                        base64_pdf = base64.b64encode(doc_bytes).decode("utf-8")
                        pdf_display = f"""<embed class="pdfobject" type="application/pdf" title="Embedded PDF" src="data:application/pdf;base64,{base64_pdf}" width="100%" height="100%">"""
                        st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    with st.expander(label=doc[0]):
                        st.markdown("Preview for given file is not available")
        else:
            st.markdown("Nothing to see here. **Upload some files...**")

        st.markdown(
            f"""

**Input Tokens** : {st.session_state.llm_tokens["Input"]} <br>
**Output Tokens** : {st.session_state.llm_tokens["Output"]}
<br><br>
**Cost** : ${(st.session_state.llm_tokens["Input"]+st.session_state.llm_tokens["Output"])*(0.2/1e6)}

""",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
