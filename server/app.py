from fastapi import FastAPI
from openenv.core.env_server import HTTPEnvServer
from .environment import ERTriageEnvironment
from models import TriageAction, Observation
import uvicorn

def create_app() -> FastAPI:
    app = FastAPI(title="ER Triage OpenEnv")
    server = HTTPEnvServer(
        env=ERTriageEnvironment,
        action_cls=TriageAction,
        observation_cls=Observation,
    )
    server.register_routes(app)

    @app.get("/")
    def root():
        return {"message": "ER Triage OpenEnv API", "status": "ok"}

    return app

def main() -> None:
    uvicorn.run("server.app:create_app", host="0.0.0.0", port=7860, factory=True)

if __name__ == "__main__":
    main()