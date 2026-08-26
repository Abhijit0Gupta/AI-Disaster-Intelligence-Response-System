# Project Rules and Team Workflow

## 1. General Rule

All team members must follow the project architecture and API contract before starting development.

Do not independently change the project structure, API field names, or module interfaces without informing the Technical Lead.

---

## 2. Folder Ownership

### Technical Lead
Responsible for:
- Overall project architecture
- ML model
- Priority logic
- Resource allocation logic
- Final integration

Main folders:
- ml/
- docs/

### Research and Data Member
Responsible for:
- Problem research
- Existing solution research
- Dataset collection

Main folders:
- research/
- data/raw/

### Data Preprocessing Member
Responsible for:
- Dataset cleaning
- Data preprocessing
- Processed dataset generation

Main folders:
- notebooks/
- data/processed/

### Backend Member
Responsible for:
- FastAPI backend
- API endpoints
- Request validation
- Backend integration

Main folder:
- backend/

### Frontend Member
Responsible for:
- Streamlit dashboard
- User input interface
- Result visualization
- API communication

Main folder:
- frontend/

### Testing and Presentation Member
Responsible for:
- Test cases
- Bug reporting
- Project documentation
- SIH presentation

Main folders:
- tests/
- presentation/

---

## 3. API Rules

All frontend and backend development must follow:

docs/api_contract.md

Do not change:

- API endpoint names
- Request field names
- Response field names
- Data types

without approval from the Technical Lead.

---

## 4. GitHub Rules

Each member should work on their own branch.

Recommended branches:

- feature/research
- feature/data-preprocessing
- feature/backend
- feature/frontend
- feature/testing-presentation

The Technical Lead is responsible for reviewing and merging final changes into the main branch.

Do not directly push unfinished or experimental code to the main branch.

---

## 5. Code Rules

- Use meaningful variable and function names.
- Keep code modular.
- Do not upload unnecessary files.
- Add comments only where necessary.
- Test code before pushing.
- Include instructions if another member needs to run your module.

---

## 6. Before Submitting Work

Before submitting work, check:

1. Does the code run?
2. Does it follow the project architecture?
3. Does it use the correct input and output formats?
4. Is the code inside the correct folder?
5. Can another team member understand how to run it?

---

## 7. Communication Rule

If a member needs to change something that may affect another module, they must inform the Technical Lead before making the change.

Examples:

- Changing an API field
- Changing dataset column names
- Changing model inputs
- Changing API response format
- Adding a new major module

---

## 8. Integration Rule

Individual modules should first be tested independently.

Final integration will follow:

Frontend
    ↓
Backend
    ↓
ML Model
    ↓
Priority Engine
    ↓
Resource Allocation Engine
    ↓
Backend Response
    ↓
Frontend Dashboard