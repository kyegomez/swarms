# Swarms Platform API Keys Documentation

This document provides detailed information on managing API keys within the Swarms Platform. API keys grant programmatic access to your account and should be handled securely. Follow the guidelines below to manage your API keys safely and effectively.

---

## Table of Contents

1. [Overview](#overview)
2. [Viewing Your API Keys](#viewing-your-api-keys)
3. [Creating a New API Key](#creating-a-new-api-key)
4. [Security Guidelines](#security-guidelines)
5. [Frequently Asked Questions](#frequently-asked-questions)

---

## Overview

API keys are unique credentials that allow you to interact with the Swarms Platform programmatically. These keys enable you to make authenticated API requests to access or modify your data. **Important:** Once a secret API key is generated, it will not be displayed again. Ensure you store it securely, as it cannot be retrieved from the platform later.

---

## Viewing Your API Keys

When you navigate to the API Keys page ([https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys)), you will see a list of your API keys along with the following information:

### Key Details:

- **Name:** A label you assign to your API key to help you identify it.
- **Key:** The secret API key is only partially visible here for security reasons.
- **Created Date:** The date when the API key was generated.
- **Actions:** Options available for managing the key (e.g., deleting an API key).

---

## Creating a New API Key

To generate a new API key, follow these steps:

1. **Attach a Credit Card:**  
   Before creating a new API key, ensure that your account has a credit card attached. This is required for authentication and billing purposes.

2. **Access the API Keys Page:**  
   Navigate to [https://swarms.world/platform/api-keys](https://swarms.world/platform/api-keys).

3. **Generate a New Key:**  
   Click on the **"Create new API key"** button. The system will generate a new secret API key for your account.

4. **Store Your API Key Securely:**  
   Once generated, the full API key will be displayed only once. Copy and store it in a secure location, as it will not be displayed again.  
   **Note:** Do not share your API key with anyone or expose it in any client-side code (e.g., browser JavaScript).

---

## Security Guidelines

- **Confidentiality:**  
  Your API keys are sensitive credentials. Do not share them with anyone or include them in public repositories or client-side code.

- **Storage:**  
  Store your API keys in secure, encrypted storage. Avoid saving them in plain text files or unsecured locations.

- **Rotation:**  
  If you suspect that your API key has been compromised, immediately delete it and create a new one.

- **Access Control:**  
  Limit access to your API keys to only those systems and personnel who absolutely require it.

---

## Frequently Asked Questions

### Q1: **Why do I need a credit card attached to my account to create an API key?**  
**A:** The requirement to attach a credit card helps verify your identity and manage billing, ensuring responsible usage of the API services provided by the Swarms Platform.

### Q2: **What happens if I lose my API key?**  
**A:** If you lose your API key, you will need to generate a new one. The platform does not store the full key after its initial generation, so recovery is not possible.

### Q3: **How can I delete an API key?**  
**A:** On the API Keys page, locate the key you wish to delete and click the **"Delete"** action next to it. This will revoke the key's access immediately.

### Q4: **Can I have multiple API keys?**  
**A:** Yes, you can generate and manage multiple API keys. Use naming conventions to keep track of their usage and purpose.

---

For any further questions or issues regarding API key management, please refer to our [Help Center](https://swarms.world/help) or contact our support team.