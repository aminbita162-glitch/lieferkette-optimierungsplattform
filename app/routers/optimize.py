from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.optimization import OptimizationInput, OptimizationResult
from app.services.optimize_service import optimize as optimize_service, DomainValidationError

router = APIRouter(tags=["Optimize"])


def _api_error(code: str, message: str, *, details: dict | None = None) -> dict:
    payload = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    if details:
        payload["error"]["details"] = details
    return payload


def _bad_request(code: str, message: str, *, details: dict | None = None) -> None:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=_api_error(code, message, details=details),
    )


@router.post(
    "/optimize",
    response_model=OptimizationResult,
    responses={
        400: {
            "description": "Bad Request (domain / business validation)",
            "content": {
                "application/json": {
                    "example": {
                        "error": {
                            "code": "INVALID_INPUT",
                            "message": "service_level must be in (0, 1)",
                            "details": {"field": "service_level"},
                        }
                    }
                }
            },
        },
        422: {"description": "Validation Error (schema)"},
        500: {"description": "Internal Server Error"},
    },
)
def optimize(payload: OptimizationInput) -> OptimizationResult:
    try:
        result = optimize_service(payload.model_dump())
        return OptimizationResult(**result)
    except DomainValidationError as exc:
        _bad_request("INVALID_INPUT", str(exc))