import collections
import copy
import json
import random
random.seed(42)
import pandas as pd
from collections import defaultdict
import sys
sys.path.insert(0, '..')
#print(sys.path)
from parsers.basic_parser import UniversalParser
import numpy as np
import math
import textdistance
from zss import Node as ZSS_Node
from zss import simple_distance


class Node:
    def __init__(self, id=None, text=None, pos=None, parent_id=None, parent_rel=None):
        self.id = id
        self.text = text
        self.pos = pos
        self.parent_id = parent_id
        self.children_ids = []
        self.parent_rel = parent_rel
        self.children_rels = []

    def get_id(self):
        return self.id

    def get_text(self):
        return self.text

    def get_parent(self):
        return self.parent_id, self.parent_rel

    def get_children(self):
        return self.children_ids, self.children_rels

    def add_child(self, id, rel):
        self.children_ids.append(id), self.children_rels.append(rel)


class Tree:
    def __init__(self, sent_data):
        self.data = sent_data
        #self.tid = None
        self.sid = None
        self.text = None
        self.nodes = []
        self.root = None
        self.sum_dd = None
        self.edges = []
        self.crossing_edges = []
        self.num_nodes = None
        self.num_nodes_wo_punct = None
        self.punct = []
        self.depths = {}
        self.degrees = {}
        self.random_tree = None
        #self.sorted = []

    def calculate_sum_dd(self):
        self.num_nodes = len(self.nodes)
        self.num_nodes_wo_punct = len([n for n in self.nodes if n.id not in self.punct])
        #self.num_nodes_wo_punct = len([i for i in self.edges if i[-1] != 'punct'])
        #self.num_nodes_wo_punct = len([i for i in self.edges if i[-1] != 'punct'])
        self.sum_dd = sum([abs(a-b) for a, b, c in self.edges if a != 0 and c != 'punct'])

    def get_mdd(self):
        return self.sum_dd / (self.num_nodes_wo_punct - 1)

    def get_ndd(self):
        return abs(np.log(self.get_mdd() / math.sqrt(self.root * self.num_nodes_wo_punct)))

    def get_left_child_ratio(self):
        ratios = []
        left_count = 0
        children_count = 0
        for node in self.nodes:
            if node.id not in self.punct:
                children = [i for i in node.children_ids if i not in self.punct]
                if len(children) > 0:
                    left = [c for c in children if c < node.id]
                    ratios.append(len(left) / len(children))
        return np.mean(ratios)

    def get_k_ary(self):
        # maximal number of children
        return np.max([len([c for c in node.children_ids if c not in self.punct]) for node in self.nodes])

    def get_k_ary_normalized(self):
        return self.get_k_ary() / (self.num_nodes_wo_punct - 1)

    def get_num_leaves(self):
        return len([node for node in self.nodes if node.id not in self.punct and len(node.children_ids) == 0])

    def get_num_leaves_normalized(self):
        return self.get_num_leaves() / (self.num_nodes_wo_punct - 1)

    def get_depth_of_nodes(self):
        def bfs(root, visited, depth=0):
            visited[root-1] = True
            depths = {root: depth}
            for child in self.nodes[root-1].children_ids:
                if child not in self.punct and not visited[child-1]:
                    child_depth = bfs(child, visited, depth+1)
                    depths.update(child_depth)
            return depths
        visited = [False] * len(self.nodes)
        depths = bfs(self.root, visited, 0)
        return depths

    def get_tree_height_2(self):
        return np.max(list(self.depths.values())) + 1

    def topology_sort(self):
        def sort_util(node, visited, stack):
            visited[node-1] = True
            for i in self.nodes[node-1].children_ids:
                sort_util(i, visited, stack)
            stack.append(node)

        visited = [False] * len(self.nodes)
        stack = []

        sort_util(self.root, visited, stack)

        return stack

    def get_topo_distance(self):
        sorted = self.topology_sort()
        unsorted = [n.id for n in self.nodes]
        try:
            assert len(sorted) == len(unsorted)

        except:
            print(' '.join([n.text for n in self.nodes]))
            print([n.children_ids for n in self.nodes])
            print(self.root)
            print(sorted)
            print(unsorted)
            print(textdistance.levenshtein.distance(unsorted, sorted))
            raise ValueError

        distance = textdistance.levenshtein.distance(unsorted, sorted)
        return distance

    def get_depth_variance(self):
        return np.var(list(self.depths.values())), np.mean(list(self.depths.values()))

    def get_degrees(self):
        degrees = defaultdict(int)

        for edge in self.edges:
            if edge[-1] != 'punct' and edge[-1] != "root":
                degrees[edge[0]] += 1

        for node in self.nodes:
            if node.id not in self.punct and node.id not in degrees.keys():
                degrees[node.id] = 0

        return degrees

    def get_degree_variance(self):
        return np.var(list(self.degrees.values())), np.mean(list(self.degrees.values()))

    def get_ndegree_variance(self):
        dv = self.get_degree_variance()
        # star graph
        n = len([n for n in self.nodes if n.pos != "PUNCT"]) if exclude_punct else len(self.nodes)
        star_dv = np.var([0] * (n-1) + [n-1])
        return dv / star_dv

    def get_longest_path(self, directed=False):
        def dfs(nodes, root):
            if len([i for i in nodes[root - 1].children_ids if i not in self.punct]) == 0:
                return 0
            children_paths = []
            for child in nodes[root - 1].children_ids:
                longest = float('-inf')
                #if exclude_punct and nodes[child-1].pos == 'PUNCT':
                #    continue
                if nodes[child-1].id in self.punct:
                    continue
                if directed:
                    longest = max(longest, dfs(nodes, child) + child - root)
                else:
                    pp = dfs(nodes, child)
                    longest = max(longest, pp + abs(child - root))

                children_paths.append(longest)
            return max(children_paths)

        longest_path = dfs(self.nodes, self.root) + self.root
        return longest_path

    def get_nnum_crossing_edges(self):
        num = len(self.get_crossing_edges())
        return num / len([e for e in self.edges if e[0] != 0 and e[2] != "punct"])

    def get_num_crossing_edges(self):
        return len(self.get_crossing_edges())

    def get_crossing_edges(self):
        crossed = []
        for i in range(len(self.edges)):
            a1 = np.min(self.edges[i][:2])
            a2 = np.max(self.edges[i][:2])

            if self.edges[i][-1] != 'punct':
                for j in range(i+1, len(self.edges)):
                    b1 = np.min(self.edges[j][:2])
                    b2 = np.max(self.edges[j][:2])
                    if self.edges[j][-1] != 'punct':
                        if a1 < b1 < a2 < b2 or b1 < a1 < b2 < a2:
                            crossed.append((self.edges[i], self.edges[j]))
        return crossed

    def is_cyclic(self):
        def is_cyclic_util(node, visited, stack):
            visited[node-1] = True
            stack[node-1] = True

            for child in self.nodes[node-1].children_ids:
                if not visited[child-1]:
                    if is_cyclic_util(child, visited, stack):
                        return True
                elif stack[child-1]:
                    return True

            stack[node-1] = False
            return False

        visited = [False] * len(self.nodes)
        stack = [False] * len(self.nodes)

        for node in self.nodes:
            if is_cyclic_util(node.id, visited, stack):
                return True
        return False

    def tree_edit_distance(self):
        def generate_random_tree():
            topology_sorted = [i+1 for i in range(len(self.nodes))]
            random.shuffle(topology_sorted)
            random_nodes = [-1] * len(self.nodes)
            random_root = topology_sorted[-1]

            for i, node_id in enumerate(topology_sorted):
                if node_id == random_root:
                    head = 0
                else:
                    head = random.choice(topology_sorted[i+1:])
                # self.nodes.append(Node(int(id), text, pos, int(head_id), rel))
                # self.edges.append((int(head_id), int(id), rel))
                node = Node(node_id, "", "", head, "")
                random_nodes[node_id-1] = node

            for node in random_nodes:
                if node.get_parent()[0] - 1 >= 0:
                    random_nodes[node.get_parent()[0] - 1].add_child(node.id, node.parent_rel)

            return random_nodes, random_root

        def build_tree(node, tree, graph):
            if len(graph[node-1].children_ids) == 0:
                return f"ZSS_Node({node})"
            tree += f"ZSS_Node({node})"
            subtrees = []
            for child in graph[node-1].children_ids:
                subtree = f".addkid({build_tree(child, '', graph)})"
                subtrees.append(subtree)
            tree += ''.join(subtrees)
            return tree

        tree = build_tree(self.root, "", self.nodes)
        tree = f"(\n{tree}\n)"
        tree = eval(tree)

        random_nodes, random_root = generate_random_tree()
        random_tree = build_tree(random_root, "", random_nodes)
        random_tree = f"(\n{random_tree}\n)"
        self.random_tree = random_tree
        random_tree = eval(random_tree)

        return simple_distance(tree, random_tree)
        #return simple_distance(random_tree, tree)

    def build_tree(self):
        #tokens = []
        root = []
        #count = 0
        for line in self.data.split("\n"):
            if line.startswith("#") or line.startswith("text = ") or line.startswith("text_id =") or len(line.split('\t')) != 10:
                continue
            else:
                try:
                    id, text, _, _, pos, _, head_id, rel, _, _ = line.split('\t')
                except:
                    print("data:\n", self.data)
                    print("error line: ", line)
                    raise ValueError
                if not id.isdigit():
                    continue
                if int(id) == int(head_id):
                    print('self-pointing detected.')
                    return False

                self.nodes.append(Node(int(id), text, pos, int(head_id), rel))
                self.edges.append((int(head_id), int(id), rel))
                if rel == 'punct':
                    self.punct.append(int(id))
                if head_id == "0":
                    #self.root = int(id)
                    root.append(int(id))
        #print('conllu:\n', self.data)
        assert len(self.nodes) == len(self.edges)
        if len(root) != 1 or len(self.nodes) == 0:
            print('Empty/multi roots detected or empty nodes detected.')
            return False

        if len(self.nodes) != self.nodes[-1].id:
            print(f"Duplicated predictions detected. Keep the first one.")
            kept = []
            for i in range(self.nodes[-1].id):
                #print(i+1)
                valid = [j for j, n in enumerate(self.nodes) if n.id == i+1]
                kept.append(valid[0])

            self.nodes = [n for j, n in enumerate(self.nodes) if j in kept]
            self.edges = [n for j, n in enumerate(self.edges) if j in kept]
            assert len(self.nodes) == self.nodes[-1].id


        self.root = root[0]
        # add children
        for node in self.nodes:
            if node.get_parent()[0]-1 >= 0:
                self.nodes[node.get_parent()[0]-1].add_child(node.id, node.parent_rel)

        if self.is_cyclic():
            print("Cycle detected.")
            return False

        self.calculate_sum_dd()
        self.depths = self.get_depth_of_nodes()
        self.degrees = self.get_degrees()
        return True




def load_data(conllu):
    if "conllu" in conllu:
        with open(conllu, 'r', encoding='utf8') as f:
            data = f.read()
    elif "json" in conllu:
        with open(conllu, 'r') as f:
            data = json.load(f)
        parser = UniversalParser(parser_type=None)
        data = parser.convert_to_conull(data, sentences=None)
    else:
        raise NotImplemented("Doc format not supported.")
    return [d.strip() for d in data.strip().split('\n\n')]


if __name__ == '__main__':
    from tqdm import tqdm
    from glob import glob

    DATA = 'hansard'
    #files = glob("../../data/hansard_final/parsed_v3/stanford/*.json")
    files = glob(f"../../data/{DATA}_final/parsed_v3/stanford/*.json")
    results = collections.defaultdict(list)

    exclude_punct = True

    for file in sorted(files): #[0:2]:
        decade = int(file.split('/')[-1][:4])
        #data = load_data("../../data/hansard_final/parsed_v3/stanford/1840.json")
        data = load_data(file)
        #print(len(data))
        #raise ValueError
        for sent_data in tqdm(data):
            tree = Tree(sent_data)
            built = tree.build_tree(exclude_punct=exclude_punct)
            if built:
                results['decade'].append(decade)
                results['tokens'].append("|".join([n.text for n in tree.nodes]))
                results['length'].append(len(tree.nodes))
                results["relations"].append("|".join([e[2] for e in tree.edges]))
                results["pos"].append("|".join([n.pos for n in tree.nodes]))
                #ndd = tree.get_ndd()
                results['mdd'].append(tree.get_mdd())
                results["ndd"].append(tree.get_ndd())

                '''
                print(tree.sum_dd)
                print(tree.num_nodes)
                print(tree.get_mdd())
                print(ndd)
                print(len(tree.get_crossing_edges(exclude_punct=exclude_punct)))
                print(tree.get_nnum_crossing_edges(exclude_punct=exclude_punct))
                print(tree.get_tree_height(exclude_punct=exclude_punct))
                print(tree.get_ntree_height(exclude_punct=exclude_punct))
                print(tree.get_longest_path(directed=False, exclude_punct=exclude_punct))
                #raise ValueError
                print(tree.get_longest_path(directed=True, exclude_punct=exclude_punct))

                print(tree.get_degree_variance(exclude_punct=exclude_punct))
                #print(np.var([4,0,0,3,0,0,0,1,0]))
                print(tree.get_ndegree_variance(exclude_punct=exclude_punct))
                print(sent_data)
                raise ValueError
                '''
                results["tree_height"].append(tree.get_tree_height(exclude_punct=exclude_punct))
                results["ntree_height_balanced"].append(tree.get_ntree_height(exclude_punct=exclude_punct, norm='balanced'))
                results["ntree_height_max"].append(tree.get_ntree_height(exclude_punct=exclude_punct, norm='max'))
                results['longest_path_directed'].append(tree.get_longest_path(directed=True, exclude_punct=exclude_punct))
                results['longest_path_undirected'].append(tree.get_longest_path(directed=False, exclude_punct=exclude_punct))
                results['#crossing_edges'].append(len(tree.get_crossing_edges()))
                results['n#crossing_edges'].append(tree.get_nnum_crossing_edges(exclude_punct=exclude_punct))
                results['degree_variance'].append(tree.get_degree_variance(exclude_punct=exclude_punct))
                results["ndegree_variance"].append(tree.get_ndegree_variance(exclude_punct=exclude_punct))

        #break
    #raise ValueError

    results = pd.DataFrame(results).sort_values("decade", ascending=True)
    print(results)
    #raise ValueError
    #print(results)
    #results.to_csv("../../data/hansard_final/parsed_v3/stanford/analyzed.csv", index=False)
    suffix = "_punct_excluded" if exclude_punct else ""
    results.to_csv(f"../../data/{DATA}_final/parsed_v3/stanford/analyzed{suffix}.tsv", index=False, sep='\t')
    #raise ValueError
