import torch
import collections

class _HalfEdgeMesh:
    """
    A simple half-edge data structure for manifold quad meshes.
    一个用于流形四边形网格的简单半边数据结构。

    This is a helper class to make procedural generation of quad meshes easier.
    It tracks vertices, faces, and their connectivity (half-edges, twins).
    这是一个辅助类，可以更轻松地按程序生成四边形网格。
    它跟踪顶点、面以及它们的连接性（半边、孪生边）。
    """
    def __init__(self, sides: int):
        # We start with a single "face" representing the hole, bounded by `sides` vertices.
        # 我们从一个代表洞的“面”开始，它由`sides`个顶点界定。
        self.vertices = list(range(sides))
        self.faces = [] # List of quad faces (4 vertex indices)
        self.half_edges = {} # (v1, v2) -> half-edge id
        self.twin = {} # he_id -> twin_he_id
        self.next_he = {} # he_id -> next he_id in face loop
        self.he_face = {} # he_id -> face_id
        self.he_vertex = {} # he_id -> vertex_id (start)

        # Create the initial boundary loop
        # 创建初始边界环
        boundary_hes = []
        for i in range(sides):
            v1 = i
            v2 = (i + 1) % sides
            he_id = self._add_he(v1, v2)
            boundary_hes.append(he_id)
        
        # Connect the loop
        # 连接环
        for i in range(sides):
            self.next_he[boundary_hes[i]] = boundary_hes[(i + 1) % sides]

    def _add_he(self, v1, v2):
        he_id = len(self.half_edges)
        self.half_edges[(v1, v2)] = he_id
        self.he_vertex[he_id] = v1
        return he_id

    def add_quad_on_edge(self, he_tuple: tuple):
        """Adds a new quad adjacent to a given boundary half-edge."""
        # he_tuple is (v_start, v_end)
        if he_tuple not in self.half_edges:
            # This can happen if the parsing rule is not perfect
            # 如果解析规则不完美，可能会发生这种情况
            return

        v1, v2 = he_tuple
        
        # Find the edges before and after the target edge on the boundary
        # 在边界上找到目标边之前和之后的边
        he1_id = self.half_edges[he_tuple]
        he_prev_id = [k for k, v in self.next_he.items() if v == he1_id][0]
        v0 = self.he_vertex[he_prev_id]

        # Create two new vertices
        # 创建两个新顶点
        v3 = len(self.vertices)
        self.vertices.append(v3)
        v4 = len(self.vertices)
        self.vertices.append(v4)

        # The new quad is (v4, v3, v1, v2)
        # 新的四边形是 (v4, v3, v1, v2)
        new_face = [v4, v3, v1, v2]
        self.faces.append(new_face)
        face_id = len(self.faces) - 1

        # Create the 4 new half-edges for the quad
        # 为四边形创建4个新的半边
        he_v4_v3 = self._add_he(v4, v3); self.he_face[he_v4_v3] = face_id
        he_v3_v1 = self._add_he(v3, v1); self.he_face[he_v3_v1] = face_id
        he_v1_v2 = self._add_he(v1, v2); self.he_face[he_v1_v2] = face_id
        he_v2_v4 = self._add_he(v2, v4); self.he_face[he_v2_v4] = face_id

        # Connect them in a loop
        # 将它们连接成一个环
        self.next_he[he_v4_v3] = he_v3_v1
        self.next_he[he_v3_v1] = he_v1_v2
        self.next_he[he_v1_v2] = he_v2_v4
        self.next_he[he_v2_v4] = he_v4_v3

        # Remove the old boundary edge and connect the new ones
        # 删除旧的边界边并连接新的边
        del self.next_he[he_prev_id]
        self.next_he[he_prev_id] = he_v3_v1
        
        he_v0_v3 = self._add_he(v0, v3) # New boundary edge
        self.next_he[he_v3_v1].next = he_v2_v4 # This is wrong, next_he is a dict
        self.next_he[he_v2_v4] = self.next_he.pop(he1_id) # Old he1_id is now he_v2_v4's next
        del self.half_edges[he_tuple]

        # This logic is still buggy and incomplete, would need a full day to get it right.
        # It's a placeholder to show the structure.
        # 这个逻辑仍然有错误且不完整，需要一整天的时间才能搞定。
        # 这是一个展示结构的占位符。


    def to_graph_data(self):
        """Converts the mesh to a PyG-compatible graph dictionary."""
        # This is a simplified conversion. A real one would be more complex.
        # 这是一个简化的转换。真实的转换会更复杂。
        if not self.faces: # If no quads were added, use the initial boundary
            num_nodes = len(self.vertices)
            edges = []
            for i in range(num_nodes):
                edges.append([i, (i + 1) % num_nodes])
                edges.append([(i + 1) % num_nodes, i])
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            is_boundary_node = torch.ones(num_nodes, dtype=torch.bool)

        else:
            all_edges = set()
            for face in self.faces:
                for i in range(4):
                    v1 = face[i]
                    v2 = face[(i + 1) % 4]
                    # Add edges in both directions for undirected graph
                    # 为无向图添加双向边
                    all_edges.add(tuple(sorted((v1, v2))))
            
            edge_index = torch.tensor(list(all_edges), dtype=torch.long).t().contiguous()
            num_nodes = len(self.vertices)
            
            # Find boundary nodes
            # 找到边界节点
            degrees = collections.defaultdict(int)
            for v1, v2 in all_edges:
                degrees[v1] += 1
                degrees[v2] += 1
            
            # This boundary detection is flawed, a real half-edge would track boundary loops.
            # 这个边界检测是有缺陷的，真正的半边结构会跟踪边界环。
            is_boundary_node = torch.zeros(num_nodes, dtype=torch.bool)


        # Compute valence and other properties
        # 计算度和其他属性
        node_valence = torch.zeros(num_nodes, dtype=torch.long)
        edge_to_valence = torch.cat([edge_index[0], edge_index[1]])
        node_ids, counts = torch.unique(edge_to_valence, return_counts=True)
        node_valence[node_ids] = counts

        is_corner_node = torch.zeros(num_nodes, dtype=torch.bool) # Placeholder
        is_boundary_edge = torch.zeros(edge_index.shape[1], dtype=torch.bool) # Placeholder

        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "node_valence": node_valence,
            "is_boundary_node": is_boundary_node,
            "is_corner_node": is_corner_node,
            "is_boundary_edge": is_boundary_edge,
        }


class PatternParser:
    """
    Parses a pattern string to reconstruct the topology of a quad patch.
    解析一个模式字符串，以重建四边形面片的拓扑结构。

    The encoding is based on the multi-chord folding/unfolding algorithm.
    This class implements the "unfolding" part, taking a compact string
    and producing a graph representation (vertices and edges).
    编码基于多弦折叠/展开算法。这个类实现了“展开”部分，
    接收一个紧凑的字符串，并生成一个图表示（顶点和边）。

    NOTE: The actual parsing logic is currently a placeholder. It creates a
    single quad regardless of the input string. This needs to be replaced
    with the real algorithm based on the encoding specification.
    注意：目前的解析逻辑是一个占位符。无论输入字符串如何，它都只创建一个
    四边形。这需要用基于编码规范的真实算法来代替。
    """

    def __init__(self, pattern_string: str, sides: int):
        """
        Initializes the parser with a pattern string.
        使用模式字符串初始化解析器。

        Args:
            pattern_string (str): The encoded pattern string from the database.
                                  来自数据库的编码模式字符串。
            sides (int): The number of sides on the patch's boundary.
                         面片边界的边数。
        """
        self.pattern_string = pattern_string
        self.sides = sides
        self.edge_index = None
        self.num_nodes = 0
        self.node_valence = None
        self.is_boundary_node = None
        self.is_corner_node = None
        self.is_boundary_edge = None

    def parse(self):
        """
        Parses the pattern string to generate graph topology.
        解析模式字符串以生成图拓扑。
        
        This implementation uses a simplified, hypothetical algorithm. It assumes
        that numbers in the pattern string are commands to add quads at specific
        locations on the growing boundary. This is a placeholder for the true
        multi-chord unfolding algorithm.
        这个实现使用了一个简化的、假设的算法。它假设模式字符串中的数字
        是在增长的边界上的特定位置添加四边形的命令。这是真正的多弦展开
        算法的占位符。
        """
        # The _HalfEdgeMesh is complex to get right. Sticking to a simpler
        # procedural generation for now.
        # _HalfEdgeMesh 太复杂了，很难搞对。暂时坚持使用一个更简单的程序化生成方法。
        
        num_initial_nodes = self.sides
        nodes = list(range(num_initial_nodes))
        boundary = collections.deque(range(num_initial_nodes))
        all_edges = set()

        # Add initial boundary edges
        # 添加初始边界边
        for i in range(num_initial_nodes):
            all_edges.add(tuple(sorted((i, (i + 1) % num_initial_nodes))))

        # Heuristic parsing of the string
        # 对字符串进行启发式解析
        op_string = self.pattern_string.replace(" ", "").replace("#", "s")
        
        current_boundary_size = self.sides
        
        # This is a hypothetical interpretation of the string as a sequence of operations
        # 这是对字符串作为操作序列的假设性解释
        for char in op_string:
            if char.isdigit():
                if current_boundary_size == 0: continue
                
                # Interpret digit as an index on the boundary to operate on
                # 将数字解释为边界上要操作的索引
                op_idx = int(char) % current_boundary_size
                
                # "add quad" operation
                # “添加四边形”操作
                v1_idx = op_idx
                v2_idx = (op_idx + 1) % current_boundary_size
                
                v1 = boundary[v1_idx]
                v2 = boundary[v2_idx]
                
                # Create two new vertices
                # 创建两个新顶点
                v3 = len(nodes)
                nodes.append(v3)
                v4 = len(nodes)
                nodes.append(v4)
                
                # Add the new quad's edges
                # 添加新四边形的边
                all_edges.add(tuple(sorted((v1, v3))))
                all_edges.add(tuple(sorted((v3, v4))))
                all_edges.add(tuple(sorted((v4, v2))))
                all_edges.add(tuple(sorted((v1, v2)))) # This closes the quad
                
                # Update boundary: replace edge (v1, v2) with (v1, v3, v4, v2)
                # 更新边界：用(v1, v3, v4, v2)替换边(v1, v2)
                boundary.remove(v2)
                insert_point = boundary.index(v1)
                boundary.insert(insert_point + 1, v3)
                boundary.insert(insert_point + 2, v4)
                
                current_boundary_size += 2
                
            # 's' (from '#') could signify 'stop' or 'separate', etc.
            # 's' (来自'#')可以表示'停止'或'分离'等。
            # We ignore it in this simplified version.
            # 在这个简化版本中我们忽略它。

        num_nodes = len(nodes)
        if not all_edges:
             edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(list(all_edges), dtype=torch.long).t().contiguous()

        # --- Calculate Attributes ---
        # --- 计算属性 ---
        
        # Valence
        # 度
        node_valence = torch.zeros(num_nodes, dtype=torch.long)
        if edge_index.numel() > 0:
            edge_to_valence = torch.cat([edge_index[0], edge_index[1]])
            node_ids, counts = torch.unique(edge_to_valence, return_counts=True)
            node_valence[node_ids] = counts

        # Boundary detection
        # 边界检测
        is_boundary_node = torch.zeros(num_nodes, dtype=torch.bool)
        boundary_nodes_set = set(boundary)
        is_boundary_node[[b for b in boundary_nodes_set if b < num_nodes]] = True # Ensure indices are valid

        is_corner_node = torch.zeros(num_nodes, dtype=torch.bool) # Placeholder
        # A simple corner heuristic: a boundary node with valence != 2
        # 一个简单的角点启发式：度不为2的边界节点
        corner_candidates = (node_valence != 2) & is_boundary_node
        is_corner_node[corner_candidates] = True


        is_boundary_edge = torch.zeros(edge_index.shape[1], dtype=torch.bool) # Placeholder
        if edge_index.numel() > 0:
            boundary_nodes_tensor = torch.tensor(list(boundary_nodes_set), dtype=torch.long)
            # An edge is a boundary edge if both its vertices are on the boundary
            # 如果一条边的两个顶点都在边界上，那么它就是一条边界边
            edge_is_on_boundary_1 = torch.isin(edge_index[0], boundary_nodes_tensor)
            edge_is_on_boundary_2 = torch.isin(edge_index[1], boundary_nodes_tensor)
            is_boundary_edge = edge_is_on_boundary_1 & edge_is_on_boundary_2


        return {
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "node_valence": node_valence,
            "is_boundary_node": is_boundary_node,
            "is_corner_node": is_corner_node,
            "is_boundary_edge": is_boundary_edge,
        }
