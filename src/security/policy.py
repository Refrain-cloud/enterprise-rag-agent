from __future__ import annotations

ROLE_DEPT = {
    "普通员工": None,
    "HR": "HR",
    "财务": "FN",
    "销售": "SL",
    "IT": "IT",
    "产品": "PD",
    "管理层": None,
}


def normalize_role(role: str) -> str:
    return role if role in ROLE_DEPT else "普通员工"


def allowed_classifications(role: str) -> list[str]:
    role = normalize_role(role)
    if role == "管理层":
        return ["public", "internal", "confidential"]
    if ROLE_DEPT[role]:
        return ["public", "internal", "confidential"]
    return ["public", "internal"]


def can_access(role: str, meta: dict) -> bool:
    role = normalize_role(role)
    classification = meta.get("classification", "internal")
    if classification == "public":
        return True
    if role == "管理层":
        return True

    dept = ROLE_DEPT[role]
    scope = meta.get("scope", "company")
    doc_dept = meta.get("dept", "")

    if classification == "internal":
        if scope == "company":
            return True
        if scope.startswith("dept:"):
            return dept is not None and scope.split(":", 1)[1] == dept
        return True

    if classification == "confidential":
        if dept is None:
            return False
        return doc_dept == dept or scope == f"dept:{dept}"

    return False
