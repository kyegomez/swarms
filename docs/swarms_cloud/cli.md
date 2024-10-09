# Swarms Cloud CLI Documentation

Welcome to the Swarms Cloud CLI documentation. This guide will help you understand how to use the CLI to interact with the Swarms Cloud platform.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Commands](#commands)
   - [onboarding](#onboarding)
   - [help](#help)
   - [get-api-key](#get-api-key)
   - [check-login](#check-login)
   - [read-docs](#read-docs)
5. [Troubleshooting](#troubleshooting)
6. [FAQs](#faqs)
7. [Contact Support](#contact-support)

## Introduction

The Swarms Cloud CLI is a command-line interface tool that allows you to interact with the Swarms Cloud platform. It provides various commands to help you manage your account, retrieve API keys, and access documentation.

## Installation

To install the Swarms Cloud CLI, you need to have Python installed on your system. You can then install the CLI using pip:

```bash
pip install swarms-cloud-cli
```

## Usage

Once installed, you can use the CLI by typing `swarms-cloud` followed by the command you wish to execute. For example:

```bash
swarms-cloud help
```

## Commands

### onboarding

Starts the onboarding process to help you set up your account.

```bash
swarms-cloud onboarding
```

### help

Displays the help message with a list of available commands.

```bash
swarms-cloud help
```

### get-api-key

Opens the API key retrieval page in your default web browser.

```bash
swarms-cloud get-api-key
```

### check-login

Checks if you are logged in and starts the cache if necessary.

```bash
swarms-cloud check-login
```

### read-docs

Redirects you to the Swarms Cloud documentation page.

```bash
swarms-cloud read-docs
```

## Troubleshooting

If you encounter any issues while using the CLI, ensure that you have the latest version installed. You can update the CLI using:

```bash
pip install --upgrade swarms-cloud-cli
```

## FAQs

**Q: How do I retrieve my API key?**  
A: Use the `get-api-key` command to open the API key retrieval page.

**Q: What should I do if I am not logged in?**  
A: Use the `check-login` command to log in and start the cache.

## Contact Support

If you need further assistance, please contact our support team at kye@swarms.world