"""
Example usage of the SupabaseConversation class for the Swarms Framework.

This example demonstrates how to:
1. Initialize a SupabaseConversation with automatic table creation
2. Add messages of different types
3. Query and search messages
4. Export/import conversations
5. Get conversation statistics

Prerequisites:
1. Install supabase-py: pip install supabase
2. Set up a Supabase project with valid URL and API key
3. Set environment variables (table will be created automatically)

Automatic Table Creation:
The SupabaseConversation will automatically create the required table if it doesn't exist.
For optimal results, you can optionally create this RPC function in your Supabase SQL Editor:

CREATE OR REPLACE FUNCTION exec_sql(sql TEXT)
RETURNS TEXT AS $$
BEGIN
    EXECUTE sql;
    RETURN 'SUCCESS';
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

Environment Variables:
   - SUPABASE_URL: Your Supabase project URL
   - SUPABASE_KEY: Your Supabase anon/service key
"""

import os
import json
from swarms.communication.supabase_wrap import (
    SupabaseConversation, 
    MessageType, 
    SupabaseOperationError,
    SupabaseConnectionError
)
from swarms.communication.base_communication import Message


def main():
    # Load environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        print("Please create a .env file with these values or set them in your environment.")
        return

    try:
        # Initialize SupabaseConversation
        print("🚀 Initializing SupabaseConversation with automatic table creation...")
        conversation = SupabaseConversation(
            supabase_url=supabase_url,
            supabase_key=supabase_key,
            system_prompt="You are a helpful AI assistant.",
            time_enabled=True,
            enable_logging=True,
            table_name="conversations",
        )
        
        print(f"✅ Successfully initialized! Conversation ID: {conversation.get_conversation_id()}")
        print("📋 Table created automatically if it didn't exist!")
        
        # Add various types of messages
        print("\n📝 Adding messages...")
        
        # Add user message
        user_msg_id = conversation.add(
            role="user", 
            content="Hello! Can you help me understand Supabase?",
            message_type=MessageType.USER,
            metadata={"source": "example_script", "priority": "high"}
        )
        print(f"Added user message (ID: {user_msg_id})")
        
        # Add assistant message with complex content
        assistant_content = {
            "response": "Of course! Supabase is an open-source Firebase alternative with a PostgreSQL database.",
            "confidence": 0.95,
            "topics": ["database", "backend", "realtime"]
        }
        assistant_msg_id = conversation.add(
            role="assistant",
            content=assistant_content,
            message_type=MessageType.ASSISTANT,
            metadata={"model": "gpt-4", "tokens_used": 150}
        )
        print(f"Added assistant message (ID: {assistant_msg_id})")
        
        # Add system message
        system_msg_id = conversation.add(
            role="system",
            content="User is asking about Supabase features.",
            message_type=MessageType.SYSTEM
        )
        print(f"Added system message (ID: {system_msg_id})")
        
        # Batch add multiple messages
        print("\n📦 Batch adding messages...")
        batch_messages = [
            Message(
                role="user",
                content="What are the main features of Supabase?",
                message_type=MessageType.USER,
                metadata={"follow_up": True}
            ),
            Message(
                role="assistant", 
                content="Supabase provides: database, auth, realtime subscriptions, edge functions, and storage.",
                message_type=MessageType.ASSISTANT,
                metadata={"comprehensive": True}
            )
        ]
        batch_ids = conversation.batch_add(batch_messages)
        print(f"Batch added {len(batch_ids)} messages: {batch_ids}")
        
        # Get conversation as string
        print("\n💬 Current conversation:")
        print(conversation.get_str())
        
        # Search for messages
        print("\n🔍 Searching for messages containing 'Supabase':")
        search_results = conversation.search("Supabase")
        for result in search_results:
            print(f"  - ID {result['id']}: {result['role']} - {result['content'][:50]}...")
        
        # Get conversation statistics
        print("\n📊 Conversation statistics:")
        stats = conversation.get_conversation_summary()
        print(json.dumps(stats, indent=2, default=str))
        
        # Get messages by role
        print("\n👤 User messages:")
        user_messages = conversation.get_messages_by_role("user")
        for msg in user_messages:
            print(f"  - {msg['content']}")
        
        # Update a message
        print(f"\n✏️ Updating message {user_msg_id}...")
        conversation.update(
            index=str(user_msg_id),
            role="user",
            content="Hello! Can you help me understand Supabase and its key features?"
        )
        print("Message updated successfully!")
        
        # Query a specific message
        print(f"\n🔎 Querying message {assistant_msg_id}:")
        queried_msg = conversation.query(str(assistant_msg_id))
        if queried_msg:
            print(f"  Role: {queried_msg['role']}")
            print(f"  Content: {queried_msg['content']}")
            print(f"  Timestamp: {queried_msg['timestamp']}")
        
        # Export conversation
        print("\n💾 Exporting conversation...")
        conversation.export_conversation("supabase_conversation_export.yaml")
        print("Conversation exported to supabase_conversation_export.yaml")
        
        # Get conversation organized by role
        print("\n📋 Messages organized by role:")
        by_role = conversation.get_conversation_by_role_dict()
        for role, messages in by_role.items():
            print(f"  {role}: {len(messages)} messages")
        
        # Get timeline
        print("\n📅 Conversation timeline:")
        timeline = conversation.get_conversation_timeline_dict()
        for date, messages in timeline.items():
            print(f"  {date}: {len(messages)} messages")
        
        # Test delete (be careful with this in production!)
        print(f"\n🗑️ Deleting system message {system_msg_id}...")
        conversation.delete(str(system_msg_id))
        print("System message deleted successfully!")
        
        # Final message count
        final_stats = conversation.get_conversation_summary()
        print(f"\n📈 Final conversation has {final_stats['total_messages']} messages")
        
        # Start a new conversation
        print("\n🆕 Starting a new conversation...")
        new_conv_id = conversation.start_new_conversation()
        print(f"New conversation started with ID: {new_conv_id}")
        
        # Add a message to the new conversation
        conversation.add(
            role="user",
            content="This is a new conversation!",
            message_type=MessageType.USER
        )
        print("Added message to new conversation")
        
        print("\n✅ Example completed successfully!")
        
    except SupabaseConnectionError as e:
        print(f"❌ Connection error: {e}")
        print("Please check your Supabase URL and key.")
    except SupabaseOperationError as e:
        print(f"❌ Operation error: {e}")
        print("Please check your database schema and permissions.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main() 