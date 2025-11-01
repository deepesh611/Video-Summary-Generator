import os
import uuid
import shutil
import streamlit as st
from pathlib import Path
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Video Summary Generator",
    page_icon="🎬",
    layout="wide"
)

# Constants
UPLOADS_DIR = Path(__file__).parent / "uploads"
SUPPORTED_FORMATS = [".mp4", ".mkv", ".mov"]


def ensure_uploads_directory():
    """Create uploads directory if it doesn't exist."""
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def generate_run_id():
    """Generate a unique run ID using timestamp and random string."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = uuid.uuid4().hex[:8]
    return f"{timestamp}_{random_str}"


def save_uploaded_file(uploaded_file, run_id):
    """Save uploaded file to the workspace directory."""
    run_dir = UPLOADS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = run_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path, run_dir


def format_file_size(size_bytes):
    """Format file size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def cleanup_upload(run_dir_path):
    """Delete the upload directory and its contents."""
    if isinstance(run_dir_path, str):
        run_dir_path = Path(run_dir_path)
    if run_dir_path.exists() and run_dir_path.is_dir():
        try:
            shutil.rmtree(run_dir_path)
            return True
        except Exception as e:
            st.error(f"Error cleaning up: {e}")
            return False
    return False


def cleanup_old_uploads(max_age_hours=24):
    """Clean up upload directories older than max_age_hours."""
    if not UPLOADS_DIR.exists():
        return
    
    current_time = datetime.now()
    cleaned_count = 0
    
    for run_dir in UPLOADS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        
        try:
            # Extract timestamp from run_id (format: YYYYMMDD_HHMMSS_random)
            dir_name = run_dir.name
            if '_' in dir_name:
                timestamp_str = '_'.join(dir_name.split('_')[:2])
                dir_time = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                age_hours = (current_time - dir_time).total_seconds() / 3600
                
                if age_hours > max_age_hours:
                    shutil.rmtree(run_dir)
                    cleaned_count += 1
        except Exception:
            # Skip directories that don't match the expected format
            continue
    
    return cleaned_count


def main():
    st.title("🎬 Video Summary Generator")
    st.markdown("Upload a video file to generate a summary.")
    
    # Ensure uploads directory exists
    ensure_uploads_directory()
    
    # Optional: Clean up old uploads on app start (runs once per session)
    if "cleanup_initiated" not in st.session_state:
        cleanup_old_uploads(max_age_hours=24)
        st.session_state.cleanup_initiated = True
    
    # Initialize session state
    if "uploaded_file_info" not in st.session_state:
        st.session_state.uploaded_file_info = None
    if "run_id" not in st.session_state:
        st.session_state.run_id = None
    
    # File uploader
    st.subheader("Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "mkv", "mov"],
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
    )
    
    if uploaded_file is not None:
        # Clean up previous upload if a new file is being uploaded
        if st.session_state.uploaded_file_info is not None:
            prev_run_dir = st.session_state.uploaded_file_info.get("run_dir")
            if prev_run_dir:
                cleanup_upload(prev_run_dir)
            st.session_state.uploaded_file_info = None
            st.session_state.run_id = None
        
        # Generate run ID if not already generated
        if st.session_state.run_id is None:
            st.session_state.run_id = generate_run_id()
        
        # Save file if not already saved
        if st.session_state.uploaded_file_info is None:
            with st.spinner("Saving uploaded file..."):
                file_path, run_dir = save_uploaded_file(
                    uploaded_file, 
                    st.session_state.run_id
                )
                
                file_size = file_path.stat().st_size
                st.session_state.uploaded_file_info = {
                    "filename": uploaded_file.name,
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "run_dir": str(run_dir),
                    "run_id": st.session_state.run_id
                }
        
        # Display file information
        file_info = st.session_state.uploaded_file_info
        
        st.success("✅ File uploaded successfully!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**File Information:**")
            st.write(f"📄 **Filename:** {file_info['filename']}")
            st.write(f"💾 **File Size:** {format_file_size(file_info['file_size'])}")
            st.write(f"🆔 **Run ID:** `{file_info['run_id']}`")
        
        with col2:
            st.markdown("**Storage Location:**")
            st.code(file_info['file_path'], language=None)
        
        # Status/Progress area (placeholder for future pipeline steps)
        st.divider()
        st.subheader("Pipeline Status")
        
        status_container = st.container()
        with status_container:
            st.info("📋 Ready to proceed with video processing pipeline.")
            st.markdown("**Next steps:**")
            st.markdown("- Frame extraction")
            st.markdown("- Video analysis")
            st.markdown("- Summary generation")
        
        # Action buttons
        st.divider()
        col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1, 1, 1, 1])
        
        with col_btn2:
            proceed_button = st.button(
                "🚀 Proceed",
                type="primary",
                use_container_width=True
            )
            
            if proceed_button:
                st.info("Proceed button clicked! Pipeline integration coming soon.")
        
        with col_btn3:
            clear_button = st.button(
                "🗑️ Clear Upload",
                type="secondary",
                use_container_width=True
            )
            
            if clear_button:
                run_dir_path = file_info.get("run_dir")
                if cleanup_upload(run_dir_path):
                    st.session_state.uploaded_file_info = None
                    st.session_state.run_id = None
                    st.success("Upload cleared successfully!")
                    st.rerun()
                else:
                    st.error("Failed to clear upload.")
        
        # Sidebar with cleanup utilities
        with st.sidebar:
            st.header("📁 Upload Management")
            st.markdown("---")
            
            # Show current upload stats
            if file_info:
                st.markdown(f"**Current Upload:**")
                st.caption(f"Run ID: `{file_info['run_id']}`")
                st.caption(f"Size: {format_file_size(file_info['file_size'])}")
            
            st.markdown("---")
            st.markdown("**Cleanup Options:**")
            
            if st.button("🧹 Clean Old Uploads", help="Remove uploads older than 24 hours"):
                with st.spinner("Cleaning old uploads..."):
                    cleaned = cleanup_old_uploads(max_age_hours=24)
                    if cleaned > 0:
                        st.success(f"Cleaned {cleaned} old upload directory/ies.")
                    else:
                        st.info("No old uploads to clean.")
            
            if st.button("⚠️ Clear All Uploads", help="Remove all upload directories"):
                if UPLOADS_DIR.exists():
                    count = sum(1 for _ in UPLOADS_DIR.iterdir() if _.is_dir())
                    if count > 0:
                        with st.spinner(f"Clearing {count} upload directory/ies..."):
                            for run_dir in UPLOADS_DIR.iterdir():
                                if run_dir.is_dir():
                                    cleanup_upload(run_dir)
                        st.session_state.uploaded_file_info = None
                        st.session_state.run_id = None
                        st.success(f"Cleared {count} upload directory/ies.")
                        st.rerun()
                    else:
                        st.info("No uploads to clear.")
    
    else:
        # Clear session state when no file is uploaded
        if st.session_state.uploaded_file_info is not None:
            st.session_state.uploaded_file_info = None
        if st.session_state.run_id is not None:
            st.session_state.run_id = None


if __name__ == "__main__":
    main()
