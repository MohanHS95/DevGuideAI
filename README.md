# DevGuide AI

A web-based learning platform for navigating land use, zoning, and development planning in New York City.

## Project Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your environment variables:
```
FLASK_APP=app.py
FLASK_ENV=development
```

4. Run the application:
```bash
flask run
```

The application will be available at `http://localhost:5000`

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `requirements.txt`: Project dependencies
- `documentation.md`: Project documentation
- `progress.md`: Development progress tracking
- `workplan.md`: Project roadmap and implementation plan

## Features (Planned)

- Display structured learning content
- Interactive chat interface with AI guide
- RAG-powered knowledge base
- Section-based navigation
- Future testing and adaptive learning features 