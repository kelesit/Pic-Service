from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Response
from fastapi.responses import FileResponse
import uvicorn
from pathlib import Path
import logging
from datetime import datetime, timezone

from processor.image_processor import ImageProcessor
from processor.rmbg_processor import RMBGProcessor
from processor.utils import ImageUtils

app = FastAPI(title="Image Processing Service")
logger = logging.getLogger(__name__)

class ServiceConfig:
    HOST = "0.0.0.0"  
    PORT = 6006
    API_KEY = "hsyzhendeshuai"  


@app.post("/api/process/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    api_key: str = Header(..., alias="X-API-Key")
):
    if api_key != ServiceConfig.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    processor = RMBGProcessor()
    temp_dir = None

    #存在问题：异步是否需要建立uniq文件名
    try:
        temp_dir = ImageUtils.create_temp_dir()
        logger.debug(f"Created temporary directory: {temp_dir}")

        # 使用唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = temp_dir / f"upload_{timestamp}.png"
        output_path = temp_dir / f"result_{timestamp}.png"

        # 保存上传文件
        await ImageUtils.save_upload_file(image, input_path)
        logger.debug(f"Saved uploaded file to: {input_path}")

        # 验证文件是否成功保存
        if not input_path.exists():
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to save uploaded file"
            )
        
        # 处理图片
        try:
            result_path = processor.remove_background(input_path, output_path)
            logger.debug(f"Processed image saved to: {result_path}")
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to process image"
            )
        
        # 验证输出文件是否存在
        if not result_path.exists():
            raise HTTPException(
                status_code=500, 
                detail="Processed image file not found"
            )
        
        """
        目前这里采用的是将结果图片存储到内存，再发送结果的模式。
        隐患：内存占用
        之所以存储为内存发是因为
        response = FileResponse(
            path=str(result_path),
            media_type="image/png",
            filename=result_filename
        )
        是异步的，而finally块(包含清理tempdir操作)会在return 后立刻执行。
        因此可能会导致还未发送响应，文件就被删除的情况
        """
        # 读取文件内容到内存
        file_content = result_path.read_bytes()

        # 清理临时目录
        ImageUtils.cleanup_temp_dir(temp_dir)
        temp_dir = None


        return Response(
            content=file_content,
            media_type="image/png",
            headers={
                "Content-Disposition": 'attachment; filename="result_img.png"'
            }
        )
    except Exception as e:
        logger.error(f"Error in remove_background endpoint: {str(e)}")
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_dir and temp_dir.exists():
            try:
                ImageUtils.cleanup_temp_dir(temp_dir)
            except Exception as e:
                logger.error(f"Error cleaning up temporary directory: {str(e)}")

    
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "image-processor"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host=ServiceConfig.HOST,
        port=ServiceConfig.PORT,
        reload=True
    )