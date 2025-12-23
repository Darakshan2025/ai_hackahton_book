---
id: 1
title: Create Constitution For Physical AI Textbook
stage: constitution
date: 2025-12-23
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: user
command: /sp.constitution
labels: [constitution, textbook, ai, robotics]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
---

## Prompt

```
/sp.constitution

Project: AI-Native Textbook on Physical AI & Humanoid Robotics

Core principles:
- Technical accuracy grounded in robotics, AI, and control theory
- Clarity for undergraduate–graduate STEM learners
- AI-native pedagogy (agents, simulations, tool-assisted learning)
- Reproducibility and traceability of all technical claims

Key standards:
- All factual claims must be cited
- Preferred sources: peer-reviewed papers, standards, official docs
- Citation style: APA
- Original writing only (0% plagiarism)
- Diagrams, equations, and examples where applicable

Constraints:
- Tooling: Docusaurus + Spec-Kit Plus + Claude Code
- Deployment: GitHub Pages
- Modular chapter structure (course-ready)
- Clear learning objectives per chapter

Success criteria:
- Technically correct and well-cited content
- Clear progression from theory → practice
- Successfully deployed textbook
- Ready for classroom and AI-agent consumption
```

## Response snapshot

Created a comprehensive constitution file for the AI-Native Textbook on Physical AI & Humanoid Robotics project with the following core principles:
1. Technical Accuracy and Scientific Rigor
2. Educational Clarity for STEM Learners
3. AI-Native Pedagogy
4. Reproducibility and Traceability
5. Originality and Academic Integrity
6. Visual Learning Support

The constitution includes additional standards, development workflow, and governance sections, with version 1.0.0 dated 2025-12-23.

## Outcome

- ✅ Impact: Created initial project constitution with 6 core principles and governance structure
- 🧪 Tests: N/A
- 📁 Files: Updated .specify/memory/constitution.md with complete constitution
- 🔁 Next prompts: Ready to proceed with project specification and planning
- 🧠 Reflection: The constitution captures all key requirements from the user's project description

## Evaluation notes (flywheel)

- Failure modes observed: None
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A