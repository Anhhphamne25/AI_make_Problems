# api/problems.py
from fastapi import APIRouter
from AI_make_Problems.api.generate import router as generate_router
from AI_make_Problems.api.edit import router as edit_router

router = APIRouter()

router.include_router(generate_router)
router.include_router(edit_router)
