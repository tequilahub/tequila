# from __future__ import annotations

# import dis
# from dataclasses import dataclass, field
# import re

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# from matplotlib.patches import FancyBboxPatch
# from tequila.utils import JoinedTransformation


# # ── Public graph API ──────────────────────────────────────────────────────────

# def walk_objectives(obj, _depth: int = 0, _id_map: dict | None = None) -> dict:
#     if _id_map is None:
#         _id_map = {}

#     counters = {"E": 0, "Bk": 0}
#     return _walk(obj, _depth, _id_map, counters)


# def diff_graphs(before: dict, after: dict) -> dict:
#     before_idx = _index_by_id(before)
#     after_idx = _index_by_id(after)
#     return _diff_node(before_idx, after_idx, after)


# def print_graph(node: dict, indent: int = 0) -> None:
#     prefix = " " * indent
#     kind = node.get("kind", "?")
#     label = node.get("label", "")
#     expr = node.get("expr", "")
#     diff = node.get("diff", "")
#     diff_s = f"[{diff}]" if diff and diff != "same" else ""

#     print(f"{prefix}[{kind}]{diff_s} {label}  {expr}")

#     t = node.get("transform")
#     if t:
#         print(f"{prefix}  transform:")
#         _print_transform(t, indent + 2)

#     for child in node.get("args", []):
#         print_graph(child, indent + 2)


# # ── Objective walker ──────────────────────────────────────────────────────────

# def _walk(obj, depth: int, id_map: dict, counters: dict) -> dict:
#     from tequila.objective.objective import (
#         Objective,
#         ExpectationValueImpl,
#         Variable,
#         FixedVariable,
#         BraKetImpl,
#     )

#     obj_id = _stable_id(obj)

#     if obj_id in id_map:
#         cached = id_map[obj_id].copy()
#         cached["_ref"] = True
#         return cached

#     node: dict = {
#         "id": obj_id,
#         "depth": depth,
#         "args": [],
#         "transform": None,
#         "meta": {},
#     }

#     id_map[obj_id] = node

#     # Constants first
#     if isinstance(obj, FixedVariable):
#         value = float(obj)
#         node.update({
#             "kind": "fixed",
#             "label": f"{value:g}",
#             "expr": f"{value:g}",
#         })

#     elif isinstance(obj, (int, float, np.integer, np.floating)):
#         value = float(obj)
#         node.update({
#             "kind": "fixed",
#             "label": f"{value:g}",
#             "expr": f"{value:g}",
#         })

#     elif isinstance(obj, Variable):
#         node.update({
#             "kind": "variable",
#             "label": str(obj.name),
#             "expr": str(obj.name),
#         })

#     # BraKet must be a leaf, otherwise QCircuits appear
#     elif isinstance(obj, BraKetImpl):
#         counters["Bk"] += 1
#         idx = counters["Bk"]

#         node.update({
#             "kind": "braket",
#             "label": f"Bk{idx}",
#             "expr": f"Bk{idx}",
#             "args": [],
#             "transform": None,
#             "meta": {"index": idx},
#         })

#     elif isinstance(obj, ExpectationValueImpl):
#         _fill_evimpl(node, obj, depth, counters)

#     elif isinstance(obj, Objective):
#         _fill_objective(node, obj, depth, id_map, counters)

#     else:
#         node.update({
#             "kind": "ignored",
#             "label": "",
#             "expr": "",
#             "args": [],
#             "transform": None,
#         })

#     return node


# def _fill_objective(node: dict, obj, depth: int, id_map: dict, counters: dict) -> None:
#     n_args = len(obj.args)
#     arg_nodes = [_walk(a, depth + 1, id_map, counters) for a in obj.args]
#     tf_node = _walk_transform(obj._transformation, obj.args, depth, id_map, counters)
    
#     expr_str = str(obj)

#     if (
#         tf_node
#         and tf_node.get("kind") == "joined"
#         and tf_node.get("meta", {}).get("op_name") == "+"
#         and "**" in expr_str
#         and len(arg_nodes) == 2
#         and all(a.get("kind") not in ("fixed",) for a in arg_nodes)  # ← guard added
#     ):
#         m = re.search(r"\*\*\s*([0-9.]+)", expr_str)
#         exponent = m.group(1) if m else "2"

#         node.update({
#             "kind": "transform",
#             "label": "+",
#             "expr": "",
#             "args": [
#                 arg_nodes[0],
#                 {
#                     "kind": "transform",
#                     "label": "^",
#                     "expr": "",
#                     "args": [
#                         arg_nodes[1],
#                         {
#                             "kind": "fixed",
#                             "label": exponent,
#                             "expr": exponent,
#                             "args": [],
#                             "transform": None,
#                             "meta": {},
#                         },
#                     ],
#                     "transform": None,
#                     "meta": {},
#                 },
#             ],
#             "transform": None,
#             "meta": {},
#         })
#         return

#     node.update({
#         "kind": "objective",
#         "label": "Objective",
#         "expr": "",
#         "args": arg_nodes,
#         "transform": tf_node,
#         "meta": {"n_args": n_args, "is_leaf": n_args == 0},
#     })


# def _fill_evimpl(node: dict, obj, depth: int, counters: dict) -> None:
#     counters["E"] += 1
#     idx = counters["E"]

#     node.update({
#         "kind": "evimpl",
#         "label": f"E{idx}",
#         "expr": f"E{idx}",
#         "meta": {"index": idx},
#     })


# # ── Transform walking ─────────────────────────────────────────────────────────

# def _walk_joined(tf: JoinedTransformation, args, depth: int) -> dict:
#     op_name = _op_name(tf.op)
#     split = tf.split

#     left_args = args[:split] if args else []
#     right_args = args[split:] if args else []

#     left_node = _walk_transform(tf.left, left_args, depth + 1)
#     right_node = _walk_transform(tf.right, right_args, depth + 1)

#     left_expr = left_node["expr"] if left_node else "?"
#     right_expr = right_node["expr"] if right_node else "?"

#     return {
#         "kind": "joined",
#         "label": op_name,
#         "expr": f"({left_expr} {op_name} {right_expr})",
#         "left": left_node,
#         "right": right_node,
#         "meta": {"op_name": op_name, "split": split},
#     }


# _NUMPY_OPS = {
#     "multiply",
#     "add",
#     "subtract",
#     "true_divide",
#     "power",
#     "sin",
#     "cos",
#     "exp",
#     "log",
#     "sqrt",
#     "abs",
# }

# _OPCODE_MAP = {
#     "BINARY_POWER": "**",
#     "BINARY_MULTIPLY": "*",
#     "BINARY_TRUE_DIVIDE": "/",
#     "BINARY_FLOOR_DIVIDE": "//",
#     "BINARY_ADD": "+",
#     "BINARY_SUBTRACT": "-",
#     "BINARY_MODULO": "%",
#     "UNARY_NEGATIVE": "-",
#     "UNARY_POSITIVE": "+",
# }

# OP_SYMBOL_MAP = {
#     "add": "+",
#     "subtract": "-",
#     "multiply": "*",
#     "true_divide": "/",
#     "power": "^",

#     "+": "+",
#     "-": "-",
#     "*": "*",
#     "/": "/",
#     "**": "^",
#     "^": "^",

#     "sin": "sin",
#     "cos": "cos",
#     "exp": "exp",
#     "log": "log",
#     "sqrt": "√",
#     "abs": "|x|",
# }

# def _decode_left_op_closure(tf) -> tuple:
#     from tequila.objective.objective import Objective

#     if not callable(tf) or not hasattr(tf, "__code__"):
#         return None, None, None

#     freevars = tf.__code__.co_freevars
#     if "left" not in freevars or "op" not in freevars:
#         return None, None, None

#     if not tf.__closure__:
#         return None, None, None

#     cells = dict(zip(freevars, tf.__closure__))
#     try:
#         left_val = cells["left"].cell_contents
#         op_val = cells["op"].cell_contents
#     except (KeyError, ValueError):
#         return None, None, None

#     if not isinstance(left_val, Objective):
#         return None, None, None

#     op_name = None
#     const = None
#     seen: set = set()
#     queue = [op_val]

#     while queue:
#         fn = queue.pop(0)
#         if id(fn) in seen:
#             continue
#         seen.add(id(fn))

#         if isinstance(fn, np.ufunc):
#             if fn.__name__ in _NUMPY_OPS:
#                 op_name = fn.__name__
#             continue

#         if not callable(fn) or not hasattr(fn, "__code__"):
#             continue

#         for c in fn.__code__.co_consts:
#             if isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, bool):
#                 if const is None:
#                     const = float(c)

#         if fn.__closure__:
#             for cell in fn.__closure__:
#                 try:
#                     v = cell.cell_contents
#                     if isinstance(v, np.ufunc):
#                         queue.append(v)
#                     elif callable(v) and hasattr(v, "__code__"):
#                         queue.append(v)
#                     elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
#                         if const is None:
#                             const = float(v)
#                 except ValueError:
#                     pass

#     op_symbol = OP_SYMBOL_MAP.get(op_name, op_name) if op_name else None
#     return left_val, op_symbol, const

# def _extract_lambda_info(tf) -> tuple[str | None, list]:
#     op_symbol = None
#     consts = []
#     seen = set()
#     queue = [tf]

#     while queue:
#         fn = queue.pop(0)

#         if id(fn) in seen:
#             continue
#         seen.add(id(fn))

#         if isinstance(fn, np.ufunc):
#             if fn.__name__ in _NUMPY_OPS:
#                 op_symbol = fn.__name__
#             continue

#         if not callable(fn) or not hasattr(fn, "__code__"):
#             continue

#         if op_symbol is None:
#             for name in fn.__code__.co_names:
#                 if name in _NUMPY_OPS:
#                     op_symbol = name
#                     break

#         if op_symbol is None:
#             for instr in dis.get_instructions(fn):
#                 if instr.opname in _OPCODE_MAP:
#                     op_symbol = _OPCODE_MAP[instr.opname]
#                     break
#                 if instr.opname == "BINARY_OP":
#                     op_symbol = instr.argrepr
#                     break

#         for c in fn.__code__.co_consts:
#             if isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, bool):
#                 consts.append(float(c))

#         if fn.__closure__:
#             for cell in fn.__closure__:
#                 try:
#                     v = cell.cell_contents
#                     if isinstance(v, np.ufunc):
#                         queue.append(v)
#                     elif callable(v) and hasattr(v, "__code__"):
#                         queue.append(v)
#                     elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
#                         consts.append(float(v))
#                 except ValueError:
#                     pass

#     unique_consts = []
#     seen_consts = set()
#     for c in consts:
#         if c not in seen_consts:
#             seen_consts.add(c)
#             unique_consts.append(c)

#     return op_symbol, unique_consts


# def _walk_transform(tf, args, depth: int, id_map: dict | None = None, counters: dict | None = None) -> dict | None:
#     if id_map is None:
#         id_map = {}
#     if counters is None:
#         counters = {"E": 0, "Bk": 0}

#     if tf is None:
#         return {
#             "kind": "identity",
#             "label": "identity",
#             "expr": "x",
#             "left": None,
#             "right": None,
#             "meta": {},
#         }

#     if isinstance(tf, JoinedTransformation):
#         return _walk_joined(tf, args, depth)

#     return _walk_lambda(tf, args, depth, id_map, counters)


# def _walk_lambda(tf, args, depth: int, id_map: dict | None = None, counters: dict | None = None) -> dict:
#     if id_map is None:
#         id_map = {}
#     if counters is None:
#         counters = {"E": 0, "Bk": 0}

#     nested_obj, op_symbol_from_left, const_from_left = _decode_left_op_closure(tf)

#     if nested_obj is not None:
#         op_sym = op_symbol_from_left or "*"
#         const_label = f"{const_from_left:g}" if const_from_left is not None else "?"

#         # ── Use the SAME id_map and counters so E/Bk numbering is consistent ──
#         nested_graph = _walk(nested_obj, depth + 1, id_map, counters)

#         if op_sym == "^":
#             return {
#                 "kind": "transform",
#                 "label": "^",
#                 "expr": "",
#                 "args": [],
#                 "transform": None,
#                 "meta": {
#                     "op_symbol": "^",
#                     "closure_consts": [const_from_left] if const_from_left is not None else [],
#                     "_nested_graph": nested_graph,
#                     "_nested_layout_children": "power",
#                     "_const_label": const_label,
#                 },
#             }
#         else:
#             return {
#                 "kind": "transform",
#                 "label": op_sym,
#                 "expr": "",
#                 "args": [],
#                 "transform": None,
#                 "meta": {
#                     "op_symbol": op_sym,
#                     "closure_consts": [const_from_left] if const_from_left is not None else [],
#                     "_nested_graph": nested_graph,
#                     "_nested_layout_children": "coeff_times",
#                     "_const_label": const_label,
#                 },
#             }

#     # Fallback: original lambda inspection
#     op_symbol, consts = _extract_lambda_info(tf)

#     if op_symbol:
#         op_symbol = OP_SYMBOL_MAP.get(op_symbol, op_symbol)
#         label = op_symbol
#     else:
#         label = "λ"

#     return {
#         "kind": "unary",
#         "label": label,
#         "expr": "",
#         "left": None,
#         "right": None,
#         "meta": {
#             "op_symbol": op_symbol,
#             "closure_consts": consts,
#         },
#     }

# # ── Diff helpers ──────────────────────────────────────────────────────────────

# def _index_by_id(node: dict, acc: dict | None = None) -> dict:
#     if acc is None:
#         acc = {}

#     acc[node["id"]] = node

#     for child in node.get("args", []):
#         _index_by_id(child, acc)

#     for sub in ("transform", "left", "right"):
#         if node.get(sub):
#             _index_by_id(node[sub], acc)

#     return acc


# def _diff_node(before_idx, after_idx, node):
#     nid = node["id"]

#     if nid in before_idx:
#         before_kind = before_idx[nid]["kind"]
#         diff = "same" if before_kind == node["kind"] else "changed"
#         node["diff"] = diff
#         node["diff_from"] = before_kind if diff == "changed" else None
#     else:
#         node["diff"] = "added"
#         node["diff_from"] = None

#     for child in node.get("args", []):
#         _diff_node(before_idx, after_idx, child)

#     for sub in ("transform", "left", "right"):
#         if node.get(sub):
#             _diff_node(before_idx, after_idx, node[sub])

#     before_child_ids = {c["id"] for c in before_idx.get(nid, {}).get("args", [])}
#     after_child_ids = {c["id"] for c in node.get("args", [])}

#     for rid in before_child_ids - after_child_ids:
#         removed = before_idx[rid].copy()
#         removed["diff"] = "removed"
#         node["args"].append(removed)

#     return node


# # ── String helpers ────────────────────────────────────────────────────────────

# def _expr_from_transform(tf, args) -> str:
#     if tf is None:
#         return _args_expr(args)

#     if isinstance(tf, JoinedTransformation):
#         op = _op_name(tf.op)
#         le = _expr_from_transform(tf.left, args[:tf.split])
#         re = _expr_from_transform(tf.right, args[tf.split:])
#         return f"({le} {op} {re})"

#     op_symbol, _ = _extract_lambda_info(tf)

#     if op_symbol:
#         name = OP_SYMBOL_MAP.get(op_symbol, op_symbol)
#     else:
#         name = "λ"

#     return f"{name}({_args_expr(args)})"


# def _args_expr(args) -> str:
#     from tequila.objective.objective import Variable, FixedVariable

#     parts = []

#     for i, a in enumerate(args):
#         if isinstance(a, Variable):
#             parts.append(str(a.name))
#         elif isinstance(a, FixedVariable):
#             parts.append(f"{float(a):g}")
#         elif isinstance(a, (int, float, np.integer, np.floating)):
#             parts.append(f"{float(a):g}")
#         else:
#             parts.append(f"E{i}")

#     return ", ".join(parts) if parts else "()"


# def _stable_id(obj) -> str:
#     try:
#         from tequila.simulators.simulator_base import BackendExpectationValue

#         if isinstance(obj, BackendExpectationValue):
#             return f"ExpectationValueImpl@{id(obj.abstract_expectationvalue):x}"

#     except ImportError:
#         pass

#     return f"{type(obj).__name__}@{id(obj):x}"


# def _op_name(op) -> str:
#     if op is None:
#         return "?"

#     if isinstance(op, np.ufunc):
#         return OP_SYMBOL_MAP.get(op.__name__, op.__name__)

#     name = getattr(op, "__name__", None) or getattr(op, "name", None)

#     if name:
#         return OP_SYMBOL_MAP.get(name, name)

#     return "λ"


# def _print_transform(node: dict, indent: int) -> None:
#     prefix = " " * indent
#     print(f"{prefix}[{node.get('kind', '?')}] {node.get('label', '')} -> {node.get('expr', '')}")

#     if node.get("left"):
#         _print_transform(node["left"], indent + 1)

#     if node.get("right"):
#         _print_transform(node["right"], indent + 1)


# # ── Layout ────────────────────────────────────────────────────────────────────

# @dataclass
# class LayoutNode:
#     label: str
#     kind: str
#     expr: str = ""
#     children: list["LayoutNode"] = field(default_factory=list)
#     x: float = 0.0
#     y: float = 0.0


# def _dict_to_layout(d: dict) -> LayoutNode | None:
#     kind = d.get("kind", "unknown")

#     if kind == "ignored":
#         return None

#     label = d.get("label", kind)
#     node = LayoutNode(label=label, kind=kind, expr="")

#     tf = d.get("transform")
#     args = d.get("args", [])

#     if tf is None or tf.get("kind") in (None, "identity"):
#         children = [
#             child
#             for a in args
#             if (child := _dict_to_layout(a)) is not None
#         ]
#         # ── NEW: unwrap bare objective shells ──────────────────────────────
#         # An "objective" node with no real transform and exactly one child
#         # is just a passthrough wrapper — return the child directly.
#         if kind == "objective" and len(children) == 1:
#             return children[0]
#         # ───────────────────────────────────────────────────────────────────
#         node.children = children
#         return node

#     if tf.get("kind") == "joined":
#         op = tf["meta"].get("op_name", "?")
#         split = tf["meta"].get("split", 1)

#         node.label = op
#         node.kind = "transform"

#         left = _layout_from_tf_and_args(tf.get("left"), args[:split])
#         right = _layout_from_tf_and_args(tf.get("right"), args[split:])

#         node.children = [c for c in (left, right) if c is not None]
#         return node

#     return _layout_from_tf_and_args(tf, args)


# def _layout_from_tf_and_args(tf, args) -> LayoutNode | None:
#     if not isinstance(tf, dict):
#         return None

#     meta = tf.get("meta", {})

#     # ── Nested-objective closure pattern (E**2, 0.5*E**2, etc.) ──────────────
#     nested_graph = meta.get("_nested_graph")
#     if nested_graph is not None:
#         pattern = meta.get("_nested_layout_children")
#         const_label = meta.get("_const_label", "?")
#         nested_layout = _dict_to_layout(nested_graph)
#         if pattern == "power":
#             return LayoutNode(
#                 label="^", kind="transform", expr="",
#                 children=[nested_layout, LayoutNode(label=const_label, kind="fixed", expr="")]
#                 if nested_layout else [],
#             )
#         if pattern == "coeff_times":
#             op = meta.get("op_symbol", "*")
#             return LayoutNode(
#                 label=op, kind="transform", expr="",
#                 children=[LayoutNode(label=const_label, kind="fixed", expr=""), nested_layout]
#                 if nested_layout else [],
#             )
#         # unknown pattern — just return nested
#         return nested_layout

#     # ── Standard path ─────────────────────────────────────────────────────────
#     children = [
#         child
#         for a in args
#         if (child := _dict_to_layout(a)) is not None
#     ]

#     if not children:
#         return None

#     tf_kind = tf.get("kind")

#     if tf_kind in (None, "identity"):
#         if len(children) == 1:
#             return children[0]
#         return LayoutNode(label="", kind="objective", expr="", children=children)

#     op = meta.get("op_symbol")
#     consts = meta.get("closure_consts") or []

#     if op in ("power", "**", "^"):
#         exponent = consts[-1] if consts else "2"
#         return LayoutNode(
#             label="^", kind="transform", expr="",
#             children=[
#                 children[0],
#                 LayoutNode(label=f"{exponent:g}" if isinstance(exponent, float) else str(exponent), kind="fixed", expr=""),
#             ],
#         )

#     if op is None and len(children) == 1 and consts:
#         coeff = float(consts[-1])
#         return LayoutNode(
#             label="*", kind="transform", expr="",
#             children=[LayoutNode(label=f"{coeff:g}", kind="fixed", expr=""), children[0]],
#         )

#     if op in ("+", "-", "*", "/"):
#         return LayoutNode(label=op, kind="transform", expr="", children=children)

#     if op in ("sin", "cos", "exp", "log", "sqrt", "√"):
#         return LayoutNode(label=op, kind="transform", expr="", children=children[:1])

#     return children[0]


# # ── Drawing style ─────────────────────────────────────────────────────────────

# COLORS = {
#     "objective": {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
#     "transform": {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
#     "evimpl": {"face": "#DFF4FF", "edge": "#000000", "text": "#000000"},
#     "braket": {"face": "#DFF4FF", "edge": "#000000", "text": "#000000"},
#     "variable": {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
#     "fixed": {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
#     "unknown": {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
# }

# BOX_W = 0.45
# BOX_H = 0.25
# RADIUS = 0.05

# X_SEP = 1.35
# Y_SEP = 1.15


# def _assign_xy(node: LayoutNode, depth: int, counter: list) -> None:
#     for child in node.children:
#         _assign_xy(child, depth + 1, counter)

#     if not node.children:
#         node.x = counter[0] * X_SEP
#         counter[0] += 1
#     else:
#         node.x = sum(c.x for c in node.children) / len(node.children)

#     node.y = depth * Y_SEP


# def _draw_edge(ax, parent: LayoutNode, child: LayoutNode) -> None:
#     ax.plot(
#         [parent.x, child.x],
#         [parent.y + BOX_H, child.y - BOX_H],
#         color="black",
#         lw=1.2,
#         zorder=1,
#     )


# def _draw_node(ax, node: LayoutNode) -> None:
#     c = COLORS.get(node.kind, COLORS["unknown"])

#     ax.add_patch(
#         FancyBboxPatch(
#             (node.x - BOX_W, node.y - BOX_H),
#             2 * BOX_W,
#             2 * BOX_H,
#             boxstyle=f"round,pad=0.03,rounding_size={RADIUS}",
#             linewidth=1.4,
#             edgecolor=c["edge"],
#             facecolor=c["face"],
#             zorder=3,
#         )
#     )

#     ax.text(
#         node.x,
#         node.y,
#         node.label,
#         ha="center",
#         va="center",
#         fontsize=11,
#         fontweight="bold",
#         color=c["text"],
#         fontfamily="monospace",
#         zorder=4,
#     )


# def _walk_draw(ax, node: LayoutNode) -> None:
#     for child in node.children:
#         _draw_edge(ax, node, child)
#         _walk_draw(ax, child)

#     _draw_node(ax, node)


# def _all_nodes(root: LayoutNode) -> list[LayoutNode]:
#     out = []
#     stack = [root]

#     while stack:
#         n = stack.pop()
#         out.append(n)
#         stack.extend(n.children)

#     return out


# # ── Public plotting API ───────────────────────────────────────────────────────

# def plot_objective(obj, title: str | None = None, save_path: str | None = None, show: bool = True):
#     graph = walk_objectives(obj)
#     root = _dict_to_layout(graph)

#     if root is None:
#         raise ValueError("Objective graph contains no drawable nodes.")

#     _assign_xy(root, 0, [0])

#     nodes = _all_nodes(root)
#     xs = [n.x for n in nodes]
#     ys = [n.y for n in nodes]

#     margin = 1.6

#     fig, ax = plt.subplots(
#         figsize=(
#             max(5, (max(xs) - min(xs) + 4) * 0.9),
#             max(3, (max(ys) - min(ys) + 3) * 0.9),
#         )
#     )

#     fig.patch.set_facecolor("white")
#     ax.set_facecolor("white")

#     ax.set_xlim(min(xs) - margin, max(xs) + margin)
#     ax.set_ylim(min(ys) - margin, max(ys) + margin)
#     ax.set_aspect("equal")
#     ax.axis("off")

#     _walk_draw(ax, root)

#     if title:
#         ax.set_title(title, fontsize=12, fontfamily="monospace", pad=10, color="black")

#     fig.tight_layout()

#     if save_path:
#         fig.savefig(save_path, dpi=150, bbox_inches="tight")
#         print(f"Saved → {save_path}")

#     if show:
#         plt.show()

#     return fig

from __future__ import annotations

import dis
from dataclasses import dataclass, field
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from tequila.utils import JoinedTransformation


# ── Public graph API ──────────────────────────────────────────────────────────

def walk_objectives(obj, _depth: int = 0, _id_map: dict | None = None) -> dict:
    if _id_map is None:
        _id_map = {}
    counters = {"E": 0, "Bk": 0}
    return _walk(obj, _depth, _id_map, counters)


def diff_graphs(before: dict, after: dict) -> dict:
    before_idx = _index_by_id(before)
    after_idx = _index_by_id(after)
    return _diff_node(before_idx, after_idx, after)


def print_graph(node: dict, indent: int = 0) -> None:
    prefix = " " * indent
    kind = node.get("kind", "?")
    label = node.get("label", "")
    expr = node.get("expr", "")
    diff = node.get("diff", "")
    diff_s = f"[{diff}]" if diff and diff != "same" else ""
    print(f"{prefix}[{kind}]{diff_s} {label}  {expr}")
    t = node.get("transform")
    if t:
        print(f"{prefix}  transform:")
        _print_transform(t, indent + 2)
    for child in node.get("args", []):
        print_graph(child, indent + 2)


# ── Objective walker ──────────────────────────────────────────────────────────

def _walk(obj, depth: int, id_map: dict, counters: dict) -> dict:
    from tequila.objective.objective import (
        Objective,
        ExpectationValueImpl,
        Variable,
        FixedVariable,
    )
    try:
        from tequila.objective.objective import BraKetImpl
    except ImportError:
        BraKetImpl = type(None)  # never matches

    obj_id = _stable_id(obj)

    if obj_id in id_map:
        cached = id_map[obj_id].copy()
        cached["_ref"] = True
        return cached

    node: dict = {
        "id": obj_id,
        "depth": depth,
        "args": [],
        "transform": None,
        "meta": {},
    }
    id_map[obj_id] = node

    if isinstance(obj, FixedVariable):
        value = float(obj)
        node.update({"kind": "fixed", "label": f"{value:g}", "expr": f"{value:g}"})

    elif isinstance(obj, (int, float, np.integer, np.floating)):
        value = float(obj)
        node.update({"kind": "fixed", "label": f"{value:g}", "expr": f"{value:g}"})

    elif isinstance(obj, Variable):
        node.update({"kind": "variable", "label": str(obj.name), "expr": str(obj.name)})

    elif isinstance(obj, BraKetImpl):
        counters["Bk"] += 1
        idx = counters["Bk"]
        node.update({
            "kind": "braket", "label": f"Bk{idx}", "expr": f"Bk{idx}",
            "args": [], "transform": None, "meta": {"index": idx},
        })

    elif isinstance(obj, ExpectationValueImpl):
        _fill_evimpl(node, obj, depth, counters)

    elif isinstance(obj, Objective):
        _fill_objective(node, obj, depth, id_map, counters)

    else:
        node.update({"kind": "ignored", "label": "", "expr": "", "args": [], "transform": None})

    return node


def _fill_objective(node: dict, obj, depth: int, id_map: dict, counters: dict) -> None:
    n_args = len(obj.args)
    arg_nodes = [_walk(a, depth + 1, id_map, counters) for a in obj.args]
    tf_node = _walk_transform(obj._transformation, obj.args, depth, id_map, counters)

    node.update({
        "kind": "objective",
        "label": "Objective",
        "expr": "",
        "args": arg_nodes,
        "transform": tf_node,
        "meta": {"n_args": n_args, "is_leaf": n_args == 0},
    })


def _fill_evimpl(node: dict, obj, depth: int, counters: dict) -> None:
    counters["E"] += 1
    idx = counters["E"]
    node.update({"kind": "evimpl", "label": f"E{idx}", "expr": f"E{idx}", "meta": {"index": idx}})


# ── Transform walking ─────────────────────────────────────────────────────────

_NUMPY_OPS = {
    "multiply", "add", "subtract", "true_divide", "power",
    "sin", "cos", "exp", "log", "sqrt", "abs",
}

_OPCODE_MAP = {
    "BINARY_POWER": "**",
    "BINARY_MULTIPLY": "*",
    "BINARY_TRUE_DIVIDE": "/",
    "BINARY_FLOOR_DIVIDE": "//",
    "BINARY_ADD": "+",
    "BINARY_SUBTRACT": "-",
    "BINARY_MODULO": "%",
    "UNARY_NEGATIVE": "-",
    "UNARY_POSITIVE": "+",
}

OP_SYMBOL_MAP = {
    "add": "+", "subtract": "-", "multiply": "*", "true_divide": "/", "power": "^",
    "+": "+", "-": "-", "*": "*", "/": "/", "**": "^", "^": "^",
    "sin": "sin", "cos": "cos", "exp": "exp", "log": "log", "sqrt": "√", "abs": "|x|",
}

def _detect_power_transform(tf, exponent: float | None) -> bool:
    if exponent is None:
        return False

    try:
        x = 3.0
        y = tf(x)
        return np.isclose(float(y), x ** exponent)
    except Exception:
        return False


def _walk_transform(tf, args, depth: int, id_map: dict | None = None, counters: dict | None = None) -> dict | None:
    if id_map is None:
        id_map = {}
    if counters is None:
        counters = {"E": 0, "Bk": 0}

    if tf is None:
        return {"kind": "identity", "label": "identity", "expr": "x", "left": None, "right": None, "meta": {}}

    if isinstance(tf, JoinedTransformation):
        return _walk_joined(tf, args, depth, id_map, counters)

    return _walk_lambda(tf, args, depth, id_map, counters)


def _walk_joined(tf: JoinedTransformation, args, depth: int, id_map: dict, counters: dict) -> dict:
    op_name = _op_name(tf.op)
    split = tf.split
    left_args = args[:split] if args else []
    right_args = args[split:] if args else []
    left_node = _walk_transform(tf.left, left_args, depth + 1, id_map, counters)
    right_node = _walk_transform(tf.right, right_args, depth + 1, id_map, counters)
    left_expr = left_node["expr"] if left_node else "?"
    right_expr = right_node["expr"] if right_node else "?"
    return {
        "kind": "joined",
        "label": op_name,
        "expr": f"({left_expr} {op_name} {right_expr})",
        "left": left_node,
        "right": right_node,
        "meta": {"op_name": op_name, "split": split},
    }


def _decode_left_op_closure(tf) -> tuple:
    """Detect the (left=Objective, op=fn) closure pattern Tequila uses for
    expressions like E**2, 0.5*E**2. Returns (nested_obj, op_symbol, const)."""
    from tequila.objective.objective import Objective

    if not callable(tf) or not hasattr(tf, "__code__"):
        return None, None, None

    freevars = tf.__code__.co_freevars
    if "left" not in freevars or "op" not in freevars:
        return None, None, None

    if not tf.__closure__:
        return None, None, None

    cells = dict(zip(freevars, tf.__closure__))
    try:
        left_val = cells["left"].cell_contents
        op_val = cells["op"].cell_contents
    except (KeyError, ValueError):
        return None, None, None

    if not isinstance(left_val, Objective):
        return None, None, None

    # Walk op_val's closure chain to find the ufunc name and numeric constant
    op_name = None
    const = None
    seen: set = set()
    queue = [op_val]

    while queue:
        fn = queue.pop(0)
        if id(fn) in seen:
            continue
        seen.add(id(fn))

        if isinstance(fn, np.ufunc):
            if fn.__name__ in _NUMPY_OPS:
                op_name = fn.__name__
            continue

        if not callable(fn) or not hasattr(fn, "__code__"):
            continue

        for c in fn.__code__.co_consts:
            if isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, bool):
                if const is None:
                    const = float(c)

        if fn.__closure__:
            for cell in fn.__closure__:
                try:
                    v = cell.cell_contents
                    if isinstance(v, np.ufunc):
                        queue.append(v)
                    elif callable(v) and hasattr(v, "__code__"):
                        queue.append(v)
                    elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                        if const is None:
                            const = float(v)
                except ValueError:
                    pass

    op_symbol = OP_SYMBOL_MAP.get(op_name, op_name) if op_name else None
    return left_val, op_symbol, const


def _extract_lambda_info(tf) -> tuple[str | None, list]:
    op_symbol = None
    consts = []
    seen = set()
    queue = [tf]

    while queue:
        fn = queue.pop(0)
        if id(fn) in seen:
            continue
        seen.add(id(fn))

        if isinstance(fn, np.ufunc):
            if fn.__name__ in _NUMPY_OPS:
                op_symbol = fn.__name__
            continue

        if not callable(fn) or not hasattr(fn, "__code__"):
            continue

        if op_symbol is None:
            for name in fn.__code__.co_names:
                if name in _NUMPY_OPS:
                    op_symbol = name
                    break

        if op_symbol is None:
            for instr in dis.get_instructions(fn):
                if instr.opname in _OPCODE_MAP:
                    op_symbol = _OPCODE_MAP[instr.opname]
                    break
                if instr.opname == "BINARY_OP":
                    op_symbol = instr.argrepr
                    break

        for c in fn.__code__.co_consts:
            if isinstance(c, (int, float, np.integer, np.floating)) and not isinstance(c, bool):
                consts.append(float(c))

        if fn.__closure__:
            for cell in fn.__closure__:
                try:
                    v = cell.cell_contents
                    if isinstance(v, np.ufunc):
                        queue.append(v)
                    elif callable(v) and hasattr(v, "__code__"):
                        queue.append(v)
                    elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                        consts.append(float(v))
                except ValueError:
                    pass

    unique_consts = []
    seen_consts: set = set()
    for c in consts:
        if c not in seen_consts:
            seen_consts.add(c)
            unique_consts.append(c)

    return op_symbol, unique_consts


def _walk_lambda(tf, args, depth: int, id_map: dict, counters: dict) -> dict:
    nested_obj, op_sym, const = _decode_left_op_closure(tf)

    if nested_obj is not None:
        # Force-correct hidden Tequila power closures.
        if _detect_power_transform(tf, const):
            op_sym = "^"

        op_sym = op_sym or "*"
        const_label = f"{const:g}" if const is not None else "?"

        nested_graph = _walk(nested_obj, depth + 1, id_map, counters)

        if op_sym == "^":
            pattern = "power"
        elif op_sym in ("*", "multiply"):
            pattern = "coeff_times"
        else:
            pattern = "unary_nested"

        return {
            "kind": "transform",
            "label": op_sym,
            "expr": "",
            "args": [],
            "transform": None,
            "meta": {
                "op_symbol": op_sym,
                "closure_consts": [const] if const is not None else [],
                "_nested_graph": nested_graph,
                "_nested_layout_children": pattern,
                "_const_label": const_label,
            },
        }

    op_symbol, consts = _extract_lambda_info(tf)
    if op_symbol:
        op_symbol = OP_SYMBOL_MAP.get(op_symbol, op_symbol)

    return {
        "kind": "unary",
        "label": op_symbol or "λ",
        "expr": "",
        "left": None,
        "right": None,
        "meta": {
            "op_symbol": op_symbol,
            "closure_consts": consts,
        },
    }

# ── Diff helpers ──────────────────────────────────────────────────────────────

def _index_by_id(node: dict, acc: dict | None = None) -> dict:
    if acc is None:
        acc = {}
    acc[node["id"]] = node
    for child in node.get("args", []):
        _index_by_id(child, acc)
    for sub in ("transform", "left", "right"):
        if node.get(sub):
            _index_by_id(node[sub], acc)
    return acc


def _diff_node(before_idx, after_idx, node):
    nid = node["id"]
    if nid in before_idx:
        before_kind = before_idx[nid]["kind"]
        diff = "same" if before_kind == node["kind"] else "changed"
        node["diff"] = diff
        node["diff_from"] = before_kind if diff == "changed" else None
    else:
        node["diff"] = "added"
        node["diff_from"] = None

    for child in node.get("args", []):
        _diff_node(before_idx, after_idx, child)
    for sub in ("transform", "left", "right"):
        if node.get(sub):
            _diff_node(before_idx, after_idx, node[sub])

    before_child_ids = {c["id"] for c in before_idx.get(nid, {}).get("args", [])}
    after_child_ids = {c["id"] for c in node.get("args", [])}
    for rid in before_child_ids - after_child_ids:
        removed = before_idx[rid].copy()
        removed["diff"] = "removed"
        node["args"].append(removed)

    return node


# ── String helpers ────────────────────────────────────────────────────────────

def _expr_from_transform(tf, args) -> str:
    if tf is None:
        return _args_expr(args)
    if isinstance(tf, JoinedTransformation):
        op = _op_name(tf.op)
        le = _expr_from_transform(tf.left, args[:tf.split])
        re_ = _expr_from_transform(tf.right, args[tf.split:])
        return f"({le} {op} {re_})"
    op_symbol, _ = _extract_lambda_info(tf)
    name = OP_SYMBOL_MAP.get(op_symbol, op_symbol) if op_symbol else "λ"
    return f"{name}({_args_expr(args)})"


def _args_expr(args) -> str:
    from tequila.objective.objective import Variable, FixedVariable
    parts = []
    for i, a in enumerate(args):
        if isinstance(a, Variable):
            parts.append(str(a.name))
        elif isinstance(a, FixedVariable):
            parts.append(f"{float(a):g}")
        elif isinstance(a, (int, float, np.integer, np.floating)):
            parts.append(f"{float(a):g}")
        else:
            parts.append(f"E{i}")
    return ", ".join(parts) if parts else "()"


def _stable_id(obj) -> str:
    try:
        from tequila.simulators.simulator_base import BackendExpectationValue
        if isinstance(obj, BackendExpectationValue):
            return f"ExpectationValueImpl@{id(obj.abstract_expectationvalue):x}"
    except ImportError:
        pass
    return f"{type(obj).__name__}@{id(obj):x}"


def _op_name(op) -> str:
    if op is None:
        return "?"
    if isinstance(op, np.ufunc):
        return OP_SYMBOL_MAP.get(op.__name__, op.__name__)
    name = getattr(op, "__name__", None) or getattr(op, "name", None)
    if name:
        return OP_SYMBOL_MAP.get(name, name)
    return "λ"


def _print_transform(node: dict, indent: int) -> None:
    prefix = " " * indent
    print(f"{prefix}[{node.get('kind','?')}] {node.get('label','')} -> {node.get('expr','')}")
    if node.get("left"):
        _print_transform(node["left"], indent + 1)
    if node.get("right"):
        _print_transform(node["right"], indent + 1)


# ── Layout ────────────────────────────────────────────────────────────────────

@dataclass
class LayoutNode:
    label: str
    kind: str
    expr: str = ""
    children: list["LayoutNode"] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0


def _dict_to_layout(d: dict) -> LayoutNode | None:
    kind = d.get("kind", "unknown")
    if kind == "ignored":
        return None

    label = d.get("label", kind)
    node = LayoutNode(label=label, kind=kind, expr="")
    tf = d.get("transform")
    args = d.get("args", [])

    # No transform (or identity) — lay out children directly
    if tf is None or (isinstance(tf, dict) and tf.get("kind") in (None, "identity")):
        children = [c for a in args if (c := _dict_to_layout(a)) is not None]
        # Unwrap single-child objective shells (they are just passthrough wrappers)
        if kind == "objective" and len(children) == 1:
            return children[0]
        node.children = children
        return node

    if isinstance(tf, dict) and tf.get("kind") == "joined":
        op = tf["meta"].get("op_name", "?")
        split = tf["meta"].get("split", 1)
        node.label = op
        node.kind = "transform"

        # Build synthetic sub-nodes for each side and recurse
        left_node = _dict_to_layout({
            "kind": "objective", "label": "Objective", "expr": "",
            "args": args[:split],
            "transform": tf.get("left"),
            "meta": {},
        })
        right_node = _dict_to_layout({
            "kind": "objective", "label": "Objective", "expr": "",
            "args": args[split:],
            "transform": tf.get("right"),
            "meta": {},
        })
        node.children = [c for c in (left_node, right_node) if c is not None]
        return node

    # Everything else (unary, transform with _nested_graph, etc.)
    return _layout_from_tf_and_args(tf, args)


def _layout_from_tf_and_args(tf, args) -> LayoutNode | None:
    if not isinstance(tf, dict):
        return None

    meta = tf.get("meta", {})

    # ── Nested-objective closure pattern (_nested_graph set by _walk_lambda) ──
    nested_graph = meta.get("_nested_graph")
    if nested_graph is not None:
        pattern = meta.get("_nested_layout_children")
        const_label = meta.get("_const_label", "?")
        nested_layout = _dict_to_layout(nested_graph)
        if pattern == "power":
            return LayoutNode(
                label="^",
                kind="transform",
                expr="",
                children=(
                    [nested_layout, LayoutNode(label=const_label, kind="fixed", expr="")]
                    if nested_layout else []
                ),
            )
        if pattern == "coeff_times":
            op = meta.get("op_symbol", "*")
            return LayoutNode(
                label=op, kind="transform", expr="",
                children=(
                    [LayoutNode(label=const_label, kind="fixed", expr=""), nested_layout]
                    if nested_layout else []
                ),
            )
        # Unknown pattern — just return the nested layout
        return nested_layout

    # ── Standard unary/lambda path ────────────────────────────────────────────
    children = [c for a in args if (c := _dict_to_layout(a)) is not None]
    if not children:
        return None

    op = meta.get("op_symbol")
    consts = meta.get("closure_consts") or []

    if op in ("power", "**", "^"):
        exponent = consts[-1] if consts else 2.0
        return LayoutNode(
            label="^", kind="transform", expr="",
            children=[
                children[0],
                LayoutNode(label=f"{exponent:g}", kind="fixed", expr=""),
            ],
        )

    if op is None and len(children) == 1 and consts:
        coeff = float(consts[-1])
        return LayoutNode(
            label="*", kind="transform", expr="",
            children=[LayoutNode(label=f"{coeff:g}", kind="fixed", expr=""), children[0]],
        )

    if op in ("+", "-", "*", "/"):
        return LayoutNode(label=op, kind="transform", expr="", children=children)

    if op in ("sin", "cos", "exp", "log", "sqrt", "√"):
        return LayoutNode(label=op, kind="transform", expr="", children=children[:1])

    return children[0]


# ── Drawing style ─────────────────────────────────────────────────────────────

COLORS = {
    "objective": {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
    "transform": {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
    "evimpl":   {"face": "#DFF4FF", "edge": "#000000", "text": "#000000"},
    "braket":   {"face": "#DFF4FF", "edge": "#000000", "text": "#000000"},
    "variable": {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
    "fixed":    {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
    "unknown":  {"face": "#FFFFFF", "edge": "#000000", "text": "#000000"},
}

BOX_W  = 0.45
BOX_H  = 0.25
RADIUS = 0.05
X_SEP  = 1.35
Y_SEP  = 1.15


def _assign_xy(node: LayoutNode, depth: int, counter: list) -> None:
    for child in node.children:
        _assign_xy(child, depth + 1, counter)
    if not node.children:
        node.x = counter[0] * X_SEP
        counter[0] += 1
    else:
        node.x = sum(c.x for c in node.children) / len(node.children)
    node.y = depth * Y_SEP


def _draw_edge(ax, parent: LayoutNode, child: LayoutNode) -> None:
    ax.plot(
        [parent.x, child.x],
        [parent.y + BOX_H, child.y - BOX_H],
        color="black", lw=1.2, zorder=1,
    )


def _draw_node(ax, node: LayoutNode) -> None:
    c = COLORS.get(node.kind, COLORS["unknown"])
    ax.add_patch(FancyBboxPatch(
        (node.x - BOX_W, node.y - BOX_H), 2 * BOX_W, 2 * BOX_H,
        boxstyle=f"round,pad=0.03,rounding_size={RADIUS}",
        linewidth=1.4, edgecolor=c["edge"], facecolor=c["face"], zorder=3,
    ))
    ax.text(
        node.x, node.y, node.label,
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=c["text"], fontfamily="monospace", zorder=4,
    )


def _walk_draw(ax, node: LayoutNode) -> None:
    for child in node.children:
        _draw_edge(ax, node, child)
        _walk_draw(ax, child)
    _draw_node(ax, node)


def _all_nodes(root: LayoutNode) -> list[LayoutNode]:
    out = []
    stack = [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(n.children)
    return out


# ── Public plotting API ───────────────────────────────────────────────────────

def plot_objective(obj, title: str | None = None, save_path: str | None = None, show: bool = True):
    graph = walk_objectives(obj)
    root = _dict_to_layout(graph)

    if root is None:
        raise ValueError("Objective graph contains no drawable nodes.")

    _assign_xy(root, 0, [0])
    nodes = _all_nodes(root)
    xs = [n.x for n in nodes]
    ys = [n.y for n in nodes]
    margin = 1.6

    fig, ax = plt.subplots(figsize=(
        max(5, (max(xs) - min(xs) + 4) * 0.9),
        max(3, (max(ys) - min(ys) + 3) * 0.9),
    ))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    _walk_draw(ax, root)

    if title:
        ax.set_title(title, fontsize=12, fontfamily="monospace", pad=10, color="black")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")

    if show:
        plt.show()

    return fig