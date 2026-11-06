"""初始化本地 OpenLDAP 目录（幂等）：ou=people / ou=groups + alice / bob + finance / hr 组。

目录结构（base dn：dc=rag,dc=local，admin：cn=admin,dc=rag,dc=local / admin）：
    ou=people
        uid=alice  userPassword=alice123
        uid=bob    userPassword=bob123
    ou=groups
        cn=finance  member=uid=alice,ou=people,...
        cn=hr       member=uid=bob,ou=people,...

用法：.venv\\Scripts\\python scripts\\bootstrap_ldap.py
"""

from __future__ import annotations

import os
import sys
import time

from ldap3 import ALL, MODIFY_REPLACE, Connection, Server
from ldap3.core.exceptions import LDAPException

LDAP_URL = os.environ.get("LDAP_URL", "ldap://127.0.0.1:389")
BASE_DN = os.environ.get("LDAP_BASE_DN", "dc=rag,dc=local")
ADMIN_DN = f"cn=admin,{BASE_DN}"
ADMIN_PASSWORD = os.environ.get("LDAP_ADMIN_PASSWORD", "admin")

PEOPLE_OU = f"ou=people,{BASE_DN}"
GROUPS_OU = f"ou=groups,{BASE_DN}"

USERS = {
    "alice": {"password": "alice123", "groups": ["finance"]},
    "bob": {"password": "bob123", "groups": ["hr"]},
}
GROUPS = ("finance", "hr")


def wait_ldap_ready(timeout_s: int = 60) -> Server:
    server = Server(LDAP_URL, get_info=ALL, connect_timeout=5)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            conn = Connection(server, user=ADMIN_DN, password=ADMIN_PASSWORD, auto_bind=True)
            conn.unbind()
            return server
        except LDAPException:
            time.sleep(2)
    raise SystemExit(f"[fail] LDAP 未就绪：{LDAP_URL}")


def ensure_entry(conn: Connection, dn: str, object_classes: list[str], attributes: dict) -> None:
    """不存在则 add；已存在则按 attributes 做 MODIFY_REPLACE（幂等）。"""
    if conn.search(dn, "(objectClass=*)", search_scope="BASE"):
        if attributes:
            changes = {
                k: [(MODIFY_REPLACE, v if isinstance(v, list) else [v])]
                for k, v in attributes.items()
            }
            conn.modify(dn, changes)
        print(f"[skip] 已存在：{dn}（属性已按目标值更新）")
        return
    conn.add(dn, object_classes, attributes)
    if conn.result["result"] != 0:
        raise SystemExit(f"[fail] 创建失败 {dn}：{conn.result}")
    print(f"[ok] 已创建：{dn}")


def main() -> int:
    server = wait_ldap_ready()
    conn = Connection(server, user=ADMIN_DN, password=ADMIN_PASSWORD, auto_bind=True)

    ensure_entry(conn, PEOPLE_OU, ["organizationalUnit"], {"ou": "people"})
    ensure_entry(conn, GROUPS_OU, ["organizationalUnit"], {"ou": "groups"})

    user_dns: dict[str, str] = {}
    for uid, spec in USERS.items():
        dn = f"uid={uid},{PEOPLE_OU}"
        user_dns[uid] = dn
        ensure_entry(
            conn,
            dn,
            ["inetOrgPerson"],
            {"cn": uid, "sn": uid, "userPassword": spec["password"]},
        )

    for group in GROUPS:
        members = [user_dns[uid] for uid, spec in USERS.items() if group in spec["groups"]]
        # groupOfNames 的 member 至少一个，本目录每个组都有成员
        ensure_entry(
            conn,
            f"cn={group},{GROUPS_OU}",
            ["groupOfNames"],
            {"cn": group, "member": members},
        )

    conn.unbind()
    print("[done] LDAP 目录初始化完成：alice→finance / bob→hr")
    return 0


if __name__ == "__main__":
    sys.exit(main())
