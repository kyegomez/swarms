import tkinter as tk

# Create the main window
root = tk.Tk()
root.title("Chat Visualization")

# Create the text area for the chat
chat_area = tk.Text(root, height=20, width=60)
chat_area.pack()

# Create the input field for the user message
input_field = tk.Entry(root)
input_field.pack()

# Create the send button
send_button = tk.Button(root, text="Send")
send_button.pack()


# Define the function to send the message
def send_message():
    # Get the message from the input field
    message = input_field.get()

    # Clear the input field
    input_field.delete(0, tk.END)

    # Add the message to the chat area
    chat_area.insert(tk.END, message + "\n")

    # Scroll to the bottom of the chat area
    chat_area.see(tk.END)


# Bind the send button to the send_message function
send_button.config(command=send_message)

# Start the main loop
root.mainloop()
