from fastapi import Depends, Request
from .config import settings

def get_model(request: Request):
    return request.app.state.model

def get_http_client(request: Request):
    return request.app.state.http_client

def get_settings_dep():
    return settings