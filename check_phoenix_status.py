#!/usr/bin/env python3
"""
Check if Phoenix is running and accessible for trace export.
"""

import requests
import sys

def check_phoenix_status(port=6006):
    """Check if Phoenix is running on the specified port."""
    
    phoenix_url = f"http://localhost:{port}"
    traces_endpoint = f"{phoenix_url}/v1/traces"
    
    print(f"🔍 Checking Phoenix status on port {port}")
    print(f"   Phoenix UI: {phoenix_url}")
    print(f"   Traces endpoint: {traces_endpoint}")
    
    try:
        # Try to connect to the main Phoenix UI
        print("\n📡 Testing Phoenix UI connection...")
        response = requests.get(phoenix_url, timeout=5)
        
        if response.status_code == 200:
            print("✅ Phoenix UI is accessible")
        else:
            print(f"⚠️  Phoenix UI returned status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Phoenix UI")
        print("   Phoenix may not be running")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection to Phoenix UI timed out")
        return False
    except Exception as e:
        print(f"❌ Error connecting to Phoenix UI: {e}")
        return False
    
    try:
        # Try to connect to the traces endpoint
        print("\n📊 Testing traces endpoint...")
        # Use HEAD request to avoid sending data
        response = requests.head(traces_endpoint, timeout=5)
        
        if response.status_code in [200, 405]:  # 405 is OK for HEAD on POST endpoint
            print("✅ Traces endpoint is accessible")
            return True
        else:
            print(f"⚠️  Traces endpoint returned status code: {response.status_code}")
            return True  # Still might work for actual trace submission
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to traces endpoint")
        print("   Phoenix traces API may not be available")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection to traces endpoint timed out")
        return False
    except Exception as e:
        print(f"❌ Error connecting to traces endpoint: {e}")
        return False


def main():
    print("🔍 Phoenix Status Checker")
    print("=" * 40)
    
    # Check default Phoenix port
    phoenix_running = check_phoenix_status(6006)
    
    print("\n" + "=" * 40)
    if phoenix_running:
        print("🎉 Phoenix appears to be running!")
        print("\n💡 Next steps:")
        print("   1. Run your tau-bench agent with tracing enabled")
        print("   2. Check the Phoenix UI for traces")
        print("   3. Look for traces with your conversation data")
        print("\n🔗 Phoenix UI: http://localhost:6006")
    else:
        print("❌ Phoenix is not running or not accessible")
        print("\n🚀 To start Phoenix:")
        print("   1. Install Phoenix: pip install arize-phoenix")
        print("   2. Start Phoenix server:")
        print("      python -c \"import phoenix as px; px.launch_app()\"")
        print("   3. Or run: phoenix serve")
        print("   4. Phoenix should start on http://localhost:6006")
    
    return 0 if phoenix_running else 1


if __name__ == "__main__":
    sys.exit(main())
