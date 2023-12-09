# Shell Commands

Contributor: [Sihan Zhao](https://github.com/Sarah816)

# Terminal Tool

This tool allows you to run shell commands on a machine.

## Setup

The tool is initialized with the following parameters:

- **name_for_model**: "Terminal"
- **description_for_model**: "Run shell commands on this machine."
- **logo_url**: "https://your-app-url.com/.well-known/logo.png"
- **contact_email**: "hello@contact.com"
- **legal_info_url**: "hello@legal.com"

## Endpoint

The tool provides the following endpoint:

- **/shell_run**: Run commands and return final output. The input should be a command string.

## Function Description

- **shell_run(commands: str)**: This function runs commands and returns the final output. The commands should be a string. The function returns the output of the executed commands.