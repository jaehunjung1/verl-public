import ray

print("Testing Ray")
try:
    ray.init(address="auto")
    print("\n=== Ray Cluster Status ===")
    print(f"Number of nodes: {len(ray.nodes())}")
    for node in ray.nodes():
        print("Node: {}, Status: {}".format(node["NodeManagerHostname"], node["Alive"]))
        # print(f"Node: {node}")
    ray.shutdown()
    print("Ray initialization successful!")
except Exception as e:
    print(f"Ray initialization failed: {str(e)}")
