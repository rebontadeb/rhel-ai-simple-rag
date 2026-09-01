__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb

client = chromadb.HttpClient(host="localhost", port=8001)
client.delete_collection("rhel-ai-docs")
print("Cleared")
