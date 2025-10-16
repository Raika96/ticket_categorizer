"""
Main Entry Point for TicketCat API
Minimal script to start the REST API server

Usage:
    # Development mode (auto-reload)
    python3 main.py
    
    # Production mode
    python3 main.py --prod
    
    # Custom host/port
    python3 main.py --host 0.0.0.0 --port 5000
"""

import argparse
import uvicorn
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description='Start TicketCat API Server')
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind (default: 127.0.0.1)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind (default: 8000)'
    )
    parser.add_argument(
        '--prod',
        action='store_true',
        help='Run in production mode (no auto-reload)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of worker processes (production only)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='baseline',
        choices=['baseline', 'distilbert'],
        help='Model to preload (default: baseline)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print(" TICKETCAT - SUPPORT TICKET CLASSIFICATION API")
    print("="*70)
    print(f"\nStarting API server...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Mode: {'Production' if args.prod else 'Development'}")
    print(f"   Model: {args.model}")
    
    if not args.prod:
        print(f"   Auto-reload: Enabled")
    else:
        print(f"   Workers: {args.workers}")
    
    print(f"\n API Documentation:")
    print(f"   Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"   ReDoc: http://{args.host}:{args.port}/redoc")
    
    print(f"\n API Endpoints:")
    print(f"   Health: http://{args.host}:{args.port}/health")
    print(f"   Classify: http://{args.host}:{args.port}/classify")
    print(f"   Stats: http://{args.host}:{args.port}/stats")
    
    print("\n" + "="*70)
    
    # Set environment variable for default model
    os.environ['DEFAULT_MODEL'] = args.model
    
    # Run server
    if args.prod:
        # Production mode
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="info",
            access_log=True
        )
    else:
        # Development mode with auto-reload
        uvicorn.run(
            "src.api:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level="debug"
        )


if __name__ == "__main__":
    main()

