from pathlib import Path
import tempfile
import logging
import shutil

from fastapi import UploadFile
import aiofiles

logger = logging.getLogger(__name__)

class ImageUtils:
    @staticmethod
    def create_temp_dir():
        return Path(tempfile.mkdtemp())
    
    @staticmethod
    def cleanup_temp_dir(temp_dir: Path):
        try:
            shutil.rmtree(str(temp_dir))
        except Exception as e:
            logger.warning(f"Failed to cleanup temp diirectory {temp_dir}: {str(e)}")

    @staticmethod
    async def save_upload_file(upload_file: UploadFile, destination: Path):
        try:
            async with aiofiles.open(str(destination), "wb") as buffer:
                content = await upload_file.read()
                await buffer.write(content)
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {str(e)}")
            raise