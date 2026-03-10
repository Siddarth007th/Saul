# SaulGPT Demo Script

## Overview
- Duration: 3 to 5 minutes.
- Goal: Show RAG-powered legal assistance with source citations.

## Script

### 0:00 to 0:30: Introduction
- Action: Show the landing page.
- Narration: Welcome to SaulGPT, the specialized AI assistant for Indian Law. Built for the ImpactNexus '26 Agentathon by Team Better Call Saul, SaulGPT helps legal professionals navigate the complex landscape of Indian statutes like the BNS, IPC, and the Constitution.

### 0:30 to 1:30: Basic Query and RAG
- Action: Type "What is the punishment for murder under BNS?"
- Narration: Let's start with a basic query. We're asking about the Bharatiya Nyaya Sanhita, which recently replaced the IPC. Notice how SaulGPT retrieves the exact section, Section 103, and provides the punishment details directly from the statute.
- Point out: Source citations on the side or bottom.

### 1:30 to 2:30: Drafting Assistance
- Action: Type "Draft a bail application for theft under Section 303 BNS."
- Narration: Beyond simple retrieval, SaulGPT can help with drafting. Here, it's generating a formal bail application, incorporating the relevant sections of the BNS we just discussed. This saves hours of manual template searching.

### 2:30 to 3:30: Constitutional Query
- Action: Type "What are the fundamental rights under the Indian Constitution?"
- Narration: SaulGPT isn't just for criminal law. It has the entire Indian Constitution in its knowledge base. Here it lists the fundamental rights, citing the relevant Articles from 14 to 32.

### 3:30 to 4:00: Conclusion and Tech Stack
- Action: Show the 'About' or 'Tech Stack' section if available, or just the UI.
- Narration: SaulGPT runs entirely locally using Ollama and a fine-tuned Llama 3.1 model. This ensures data privacy for sensitive legal work. Thank you!

## Fallback Plan
- Issue: Ollama is slow or down.
- Fallback: Use pre-recorded video of the same queries.
- Issue: Database connection error.
- Fallback: Show the `data/processed/chunks.jsonl` to prove the data is there and explain the RAG logic.
