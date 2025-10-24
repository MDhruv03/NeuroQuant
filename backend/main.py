
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import uvicorn
from datetime import datetime

from backend.api import routes
from database.database import create_db_and_tables
from config import config
from utils.logging_config import setup_logger
from utils.exceptions import NeuroQuantException

# Setup logger
logger = setup_logger(__name__)

# Initialize database
logger.info("Initializing database...")
create_db_and_tables()

# Create FastAPI app
app = FastAPI(
    title=config.api.TITLE,
    version=config.api.VERSION,
    description=config.api.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.ORIGINS,
    allow_credentials=config.cors.ALLOW_CREDENTIALS,
    allow_methods=config.cors.ALLOW_METHODS,
    allow_headers=config.cors.ALLOW_HEADERS,
)


# Exception handlers
@app.exception_handler(NeuroQuantException)
async def neuroquant_exception_handler(request, exc):
    """Handle custom NeuroQuant exceptions"""
    logger.error(f"NeuroQuant exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": exc.__class__.__name__,
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors"""
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "ValidationError",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Execute on application startup"""
    logger.info("=" * 60)
    logger.info("Starting NeuroQuant Trading System")
    logger.info(f"Version: {config.api.VERSION}")
    logger.info(f"Environment: {config.development.ENVIRONMENT}")
    logger.info(f"Debug Mode: {config.development.DEBUG}")
    logger.info(f"Database: {config.database.URL}")
    logger.info(f"Sentiment Analysis: {'Enabled' if config.sentiment.ENABLED else 'Disabled'}")
    logger.info(f"Initial Portfolio: ${config.financial.INITIAL_PORTFOLIO:,.2f}")
    logger.info(f"API running on http://{config.api.HOST}:{config.api.PORT}")
    logger.info(f"API Docs: http://{config.api.HOST}:{config.api.PORT}/docs")
    logger.info("=" * 60)


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Execute on application shutdown"""
    logger.info("Shutting down NeuroQuant Trading System...")


# Include routers
app.include_router(routes.router)

if __name__ == "__main__":
    logger.info("Starting server...")
    
    uvicorn.run(
        app,
        host=config.api.HOST,
        port=config.api.PORT,
        reload=config.api.RELOAD,
        log_level=config.logging.LEVEL.lower()
    )

