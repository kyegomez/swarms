import os
from swarms.communication.supabase_wrap import SupabaseConversation, MessageType, SupabaseOperationError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "conversations" # Make sure this table exists in your Supabase DB

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY environment variables must be set.")
        print("Please create a .env file with these values or set them in your environment.")
        return

    print(f"Attempting to connect to Supabase URL: {SUPABASE_URL[:20]}...") # Print partial URL for security

    try:
        # Initialize SupabaseConversation
        print(f"\n--- Initializing SupabaseConversation for table '{TABLE_NAME}' ---")
        convo = SupabaseConversation(
            supabase_url=SUPABASE_URL,
            supabase_key=SUPABASE_KEY,
            table_name=TABLE_NAME,
            time_enabled=True, # DB schema handles timestamps by default
            enable_logging=True,
        )
        print(f"Initialized. Current Conversation ID: {convo.get_conversation_id()}")

        # --- Add messages ---
        print("\n--- Adding messages ---")
        user_msg_id = convo.add("user", "Hello, Supabase!", message_type=MessageType.USER, metadata={"source": "test_script"})
        print(f"Added user message. ID: {user_msg_id}")

        assistant_msg_content = {"response": "Hi there! How can I help you today?", "confidence": 0.95}
        assistant_msg_id = convo.add("assistant", assistant_msg_content, message_type=MessageType.ASSISTANT)
        print(f"Added assistant message. ID: {assistant_msg_id}")

        system_msg_id = convo.add("system", "Conversation started.", message_type=MessageType.SYSTEM)
        print(f"Added system message. ID: {system_msg_id}")


        # --- Display conversation ---
        print("\n--- Displaying conversation ---")
        convo.display_conversation()

        # --- Get all messages for current conversation ---
        print("\n--- Retrieving all messages for current conversation ---")
        all_messages = convo.get_messages()
        if all_messages:
            print(f"Retrieved {len(all_messages)} messages:")
            for msg in all_messages:
                print(f"  ID: {msg.get('id')}, Role: {msg.get('role')}, Content: {str(msg.get('content'))[:50]}...")
        else:
            print("No messages found.")

        # --- Query a specific message ---
        if user_msg_id:
            print(f"\n--- Querying message with ID: {user_msg_id} ---")
            queried_msg = convo.query(str(user_msg_id)) # Query expects string ID
            if queried_msg:
                print(f"Queried message: {queried_msg}")
            else:
                print(f"Message with ID {user_msg_id} not found.")

        # --- Search messages ---
        print("\n--- Searching for messages containing 'Supabase' ---")
        search_results = convo.search("Supabase")
        if search_results:
            print(f"Found {len(search_results)} matching messages:")
            for msg in search_results:
                print(f"  ID: {msg.get('id')}, Content: {str(msg.get('content'))[:50]}...")
        else:
            print("No messages found matching 'Supabase'.")

        # --- Update a message ---
        if assistant_msg_id:
             print(f"\n--- Updating message with ID: {assistant_msg_id} ---")
             new_content = {"response": "I am an updated assistant!", "confidence": 0.99}
             convo.update(index_or_id=str(assistant_msg_id), content=new_content, metadata={"updated_by": "test_script"})
             updated_msg = convo.query(str(assistant_msg_id))
             print(f"Updated message: {updated_msg}")


        # --- Get last message ---
        print("\n--- Getting last message ---")
        last_msg = convo.get_last_message_as_string()
        print(f"Last message: {last_msg}")


        # --- Export and Import (example) ---
        # Create a dummy export file name based on conversation ID
        export_filename_json = f"convo_{convo.get_conversation_id()}.json"
        export_filename_yaml = f"convo_{convo.get_conversation_id()}.yaml"
        
        print(f"\n--- Exporting conversation to {export_filename_json} and {export_filename_yaml} ---")
        convo.save_as_json_on_export = True # Test JSON export
        convo.export_conversation(export_filename_json)
        convo.save_as_json_on_export = False # Switch to YAML for next export
        convo.save_as_yaml_on_export = True
        convo.export_conversation(export_filename_yaml)


        print("\n--- Starting a new conversation and importing from JSON ---")
        new_convo_id_before_import = convo.start_new_conversation()
        print(f"New conversation started with ID: {new_convo_id_before_import}")
        convo.import_conversation(export_filename_json) # This will start another new convo internally
        print(f"Conversation imported from {export_filename_json}. Current ID: {convo.get_conversation_id()}")
        convo.display_conversation()

        # --- Delete a message ---
        if system_msg_id: # Using system_msg_id from the *original* conversation for this demo
            print(f"\n--- Attempting to delete message with ID: {system_msg_id} from a *previous* conversation (might not exist in current) ---")
            # Note: After import, system_msg_id refers to an ID from a *previous* conversation.
            # To robustly test delete, you'd query a message from the *current* imported conversation.
            # For this example, we'll just show the call.
            # Let's add a message to the *current* conversation and delete that one.
            temp_msg_id_to_delete = convo.add("system", "This message will be deleted.")
            print(f"Added temporary message with ID: {temp_msg_id_to_delete}")
            convo.delete(str(temp_msg_id_to_delete))
            print(f"Message with ID {temp_msg_id_to_delete} deleted (if it existed in current convo).")
            if convo.query(str(temp_msg_id_to_delete)) is None:
                print("Verified: Message no longer exists.")
            else:
                print("Warning: Message still exists or query failed.")


        # --- Clear current conversation ---
        print("\n--- Clearing current conversation ---")
        convo.clear()
        print(f"Conversation {convo.get_conversation_id()} cleared.")
        if not convo.get_messages():
            print("Verified: No messages in current conversation after clearing.")


        print("\n--- Example Finished ---")

    except SupabaseOperationError as e:
        print(f"Supabase Connection Error: {e}")
    except SupabaseOperationError as e:
        print(f"Supabase Operation Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()