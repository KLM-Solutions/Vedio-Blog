import warnings
warnings.filterwarnings('ignore')

import streamlit as st
from moviepy.editor import VideoFileClip
import os
from pathlib import Path
import tempfile
from openai import OpenAI


# Initialize OpenAI client with Streamlit secrets
try:
    client = OpenAI(
        api_key=st.secrets["OPENAI_API_KEY"],
        base_url="https://api.openai.com/v1"
    )
except Exception as e:
    st.error(f"Error initializing OpenAI client: {str(e)}")
    client = None

def extract_audio(video_path, output_path=None):
    """
    Extract audio from a video file and save it as MP3.
    """
    video = None
    audio = None
    try:
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if output_path is None:
            output_path = video_path.with_suffix('.mp3')
        else:
            output_path = Path(output_path)
            
        # Debug information
        st.write(f"Processing video: {video_path}")
        st.write(f"Output path: {output_path}")
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load video with explicit audio flag
        video = VideoFileClip(str(video_path), audio_buffersize=2000)
        
        if video is None:
            raise ValueError("Failed to load video file")
        
        # Extract and write audio
        if hasattr(video, 'audio') and video.audio is not None:
            audio = video.audio
            st.write("Extracting audio...")
            audio.write_audiofile(
                str(output_path),
                fps=44100,
                nbytes=2,
                buffersize=2000,
                verbose=False,
                logger=None
            )
            st.write("Audio extraction complete!")
            return str(output_path)
        else:
            raise ValueError("No audio track found in the video")
            
    except Exception as e:
        st.error(f"Error in extract_audio: {str(e)}")
        if video is None:
            st.error("Video failed to load properly")
        raise
        
    finally:
        # Cleanup
        if audio is not None:
            try:
                audio.close()
            except:
                pass
        if video is not None:
            try:
                video.close()
            except:
                pass

def transcribe_audio(audio_path):
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

def generate_blog(transcript):
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
            ]
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating blog: {str(e)}")
        raise

def main():
    st.title("Video to Blog Generator")
    st.write("Upload a video file to generate a blog post")

    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_file is not None:
        try:
            # Save uploaded file with proper handling
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file.flush()  # Ensure all data is written
                video_path = tmp_file.name
                st.write(f"Temporary file created at: {video_path}")

            if st.button("Generate Blog"):
                try:
                    # Step 1: Extract Audio
                    with st.spinner("Extracting audio..."):
                        output_path = Path(video_path).with_suffix('.mp3')
                        audio_file = extract_audio(video_path, output_path)
                        if audio_file and Path(audio_file).exists():
                            st.success("Audio extracted successfully!")
                        else:
                            raise ValueError("Audio extraction failed")

                    # Step 2: Transcribe Audio
                    with st.spinner("Transcribing audio..."):
                        with open(audio_file, "rb") as audio:
                            transcription = client.audio.transcriptions.create(
                                model="whisper-1", 
                                file=audio
                            )
                        transcript = transcription.text
                        st.success("Audio transcribed successfully!")
                        
                        with st.expander("View Transcript"):
                            st.text(transcript)

                    # Step 3: Generate Blog
                    with st.spinner("Generating blog post..."):
                        blog_content = generate_blog(transcript)
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
                    for file in [video_path, output_path]:
                        try:
                            if file and Path(file).exists():
                                os.unlink(file)
                        except Exception as e:
                            st.warning(f"Failed to cleanup file {file}: {e}")

        except Exception as e:
            st.error(f"Error processing upload: {e}")

if __name__ == "__main__":
    main() 
    
