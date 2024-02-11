import os
import datetime
import time

# Configure your repository details
repo_path = '.'
file_path = 'example.py'
commit_message = 'swarm'

def make_change_and_commit(repo_path, file_path, commit_message):
    # Change to the repository directory
    os.chdir(repo_path)
    
    # Make a change in the file
    with open(file_path, 'a') as file:
        file.write('.') # Appending a dot to the file

    # Add the file to staging
    os.system('git add ' + file_path)
    
    # Commit the change
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    os.system(f'git commit -m "{commit_message} at {current_time}"')
    
    # Push the commit
    os.system('git push')

if __name__ == "__main__":
    while True:
        make_change_and_commit(repo_path, file_path, commit_message)
        print("Commit made. Waiting 10 seconds for the next commit.")
        time.sleep(10)  # Wait for 10 seconds before the next iteration
