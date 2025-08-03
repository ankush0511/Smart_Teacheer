from youtube_utils import extract_video_id, get_transcript_and_summary
from pinecone_utils import create_pinecone_index, index_documents, retrieve_documents
from llm_utils import generate_answer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Global Pinecone Index Initialization
pinecone_index = create_pinecone_index()


def process_youtube_video(youtube_url: str) -> dict:
    """
    Extract transcript and summary from a YouTube video and index the summary in Pinecone.
    Returns transcript and summary as dictionary.
    """
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    transcript, summary = get_transcript_and_summary(video_id)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7000, chunk_overlap=50)
    documents = text_splitter.create_documents([summary])
    index_documents(documents, pinecone_index)

    return {
        "video_id": video_id,
        "transcript": transcript,
        "summary": summary
    }


def answer_query(query: str) -> str:
    """
    Answer a user question using RAG over the indexed summary.
    """
    if not query:
        raise ValueError("Query is empty")

    retrieved_docs = retrieve_documents(query, pinecone_index)
    return generate_answer(query, retrieved_docs)


# Example usage for testing or backend call:
if __name__ == "__main__":
    print("Processing YouTube video...")
    url = "https://youtu.be/DX3q_lcbT88?si=FGQfyz6BOW-z" # input("Enter YouTube URL: ")
    result = process_youtube_video(url)
    print("Summary:\n", result["summary"])
    while(True):
        user_input = input("Enter a query (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        answer = answer_query(user_input)
        print("Answer:\n", answer)
