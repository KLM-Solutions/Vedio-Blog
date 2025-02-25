import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from moviepy.editor import VideoFileClip
import os
from pathlib import Path
import tempfile
from openai import OpenAI

# Set up session state for API key
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# Try to get API key from secrets, but provide alternative if not available
try:
    default_api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    default_api_key = ""

def extract_audio(video_path, output_path=None):
    """
    Extract audio from a video file and save it as MP3.
    
    Args:
        video_path (str): Path to the input video file
        output_path (str, optional): Path for the output MP3 file. If not provided,
                                   will use the same name as video with .mp3 extension
    
    Returns:
        str: Path to the generated MP3 file
    """
    try:
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if output_path is None:
            output_path = video_path.with_suffix('.mp3')
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
            
        video = VideoFileClip(str(video_path))
        if video.audio is None:
            raise ValueError("Video has no audio track")
            
        audio = video.audio
        audio.write_audiofile(str(output_path))
        
        audio.close()
        video.close()
        
        return str(output_path)
        
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        raise
    finally:
        try:
            audio.close()
            video.close()
        except:
            pass

def transcribe_audio(audio_path, client):
    """
    Transcribe audio file to text using OpenAI's Whisper-1 API
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcription.text
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        raise

def generate_blog(transcript, client):
    """
    Generate a blog post from the transcript using GPT
    """
    try:
        prompt = f"""
        Based on the following transcript, create a well-structured blog post:
        
        Transcript:
        {transcript}
        
        Please format the blog post with:
        1. An engaging title
        2. Introduction
        3. Main points with subheadings
        4. Conclusion
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional blog writer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating blog: {str(e)}")
        raise

def main():
    st.title("Video to Blog Generator")
    st.write("Upload a video file to generate a blog post")
    
    # API key input
    api_key_input = st.text_input(
        "OpenAI API Key", 
        value=st.session_state.api_key or default_api_key,
        type="password",
        help="Enter your OpenAI API key. This is required for transcription and blog generation."
    )
    
    # Update session state
    st.session_state.api_key = api_key_input
    
    # Initialize OpenAI client with the provided API key
    client = OpenAI(api_key=st.session_state.api_key)

    # File uploader
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name

        if st.button("Generate Blog"):
            # Validate API key is provided
            if not st.session_state.api_key:
                st.error("Please enter your OpenAI API key to continue.")
                return
            try:
                # Step 1: Extract Audio
                with st.spinner("Extracting audio..."):
                    output_path = Path(video_path).with_suffix('.mp3')
                    audio_file = extract_audio(video_path, output_path)
                    st.success("Audio extracted successfully!")

                # Step 2: Transcribe Audio
                with st.spinner("Transcribing audio..."):
                    transcript = transcribe_audio(audio_file, client)
                    st.success("Audio transcribed successfully!")
                    
                    # Show transcript in expander
                    with st.expander("View Transcript"):
                        st.text(transcript)

                # Step 3: Generate Blog
                with st.spinner("Generating blog post..."):
                    blog_content = generate_blog(transcript, client)
                    st.success("Blog post generated successfully!")

                # Display blog content
                st.markdown("## Generated Blog Post")
                st.markdown(blog_content)

                # Add download button for blog content
                st.download_button(
                    label="Download Blog Post",
                    data=blog_content,
                    file_name="blog_post.md",
                    mime="text/markdown"
                )

            except Exception as e:
                st.error(f"An error occurred: {e}")

            finally:
                # Cleanup temporary files
                try:
                    os.unlink(video_path)
                    os.unlink(output_path)
                except:
                    pass

if __name__ == "__main__":
    main()
