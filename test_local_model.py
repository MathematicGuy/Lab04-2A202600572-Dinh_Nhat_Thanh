import sys
from pathlib import Path

# Add src and root to sys.path to ensure correct imports
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import os
from dotenv import load_dotenv
load_dotenv(override=True)

from core.llm import build_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

def test_local_model():
    sys.stdout.reconfigure(encoding='utf-8')
    print("🚀 Initializing test run for the local Ollama model...")
    
    try:
        # Build local Ollama chat model
        print("🔧 Compiling ChatOllama model...")
        chat_model = build_chat_model(
            provider="ollama",
            model_name="phi3",
            temperature=0.0
        )
        print("✅ ChatOllama compiled successfully!")
        
        # Simple test prompt
        messages = [
            SystemMessage(content="You are a helpful travel assistant. Keep your response very short, concise, and in Vietnamese."),
            HumanMessage(content="Tôi muốn đi du lịch Nha Trang 2 ngày.")
        ]
        
        print("\n💬 Sending test message to local Ollama model...")
        response = chat_model.invoke(messages)
        print("\n📥 Model Response:")
        print("-" * 50)
        print(response.content)
        print("-" * 50)
        print("🎉 Local Ollama model test execution complete!")
        
    except Exception as e:
        print(f"\n❌ Error during local Ollama execution: {e}")

if __name__ == "__main__":
    test_local_model()
