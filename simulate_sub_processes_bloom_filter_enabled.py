import subprocess
import json
import time
import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from node2 import Block, serialize_block

def serialize_private_key(private_key):
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')  # Convert bytes to string for passing around

# Deserialize the private key from PEM (string format)
def deserialize_private_key(private_key_str):
    return serialization.load_pem_private_key(
        private_key_str.encode('utf-8'),  # Convert string to bytes
        password=None,  # No password used
        backend=default_backend()
    )


def generate_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key

def serialize_public_key(public_key):
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode('utf-8')  # Convert to string for easy distribution

def serialize_private_key(private_key):
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    ).decode('utf-8')  # Convert to string for passing to nodes

def simulate_voting_process(num_nodes=20, block_interval=10):
        # Create the genesis block
    f = 6
    base_port = 5000

    # Prepare nodes info
    nodes_info = [{'node_id': i, 'host': 'localhost', 'port': base_port + i} for i in range(num_nodes)]

    # Generate key pairs for each node and serialize public/private keys
    node_keys = {}
    public_keys = {}
    for node_info in nodes_info:
        node_id = node_info['node_id']
        private_key, public_key = generate_key_pair()
        node_keys[node_id] = serialize_private_key(private_key)  # Store the private key for each node
        public_keys[node_id] = serialize_public_key(public_key)  # Store serialized public keys

    # Serialize the genesis block

    # Create peers mapping for each node and add public keys
    peers = {}
    for node_info in nodes_info:
        node_id = node_info['node_id']
        node_peers = {other_node['node_id']: (other_node['host'], other_node['port']) for other_node in nodes_info if other_node['node_id'] != node_id}
        peers[node_id] = node_peers

    proposer_id = 0  # You can assign any node as the proposer of the genesis block
    private_key_str = node_keys[proposer_id]  # This is the serialized private key string

    # Deserialize the private key
    private_key = deserialize_private_key(private_key_str)
    genesis_block = Block(#block_hash="genesis", 
        height=0, previous_block_hash=None, proposer_id=proposer_id)
    
    # Sign the genesis block using the proposer's private key
    genesis_block.sign_block(private_key)  # Assuming `private_key` belongs to the node initializing the genesis block
    genesis_block_json = serialize_block(genesis_block)

    # Start each node as a separate process and pass private and public keys
    processes = []
    for node_info in nodes_info:
        node_id = node_info['node_id']
        node_peers = peers[node_id]
        node_peers_json = json.dumps(node_peers)
        node_public_keys_json = json.dumps(public_keys)  # Pass all public keys
        node_private_key = node_keys[node_id]  # Pass private key

        # Launch a separate Python instance for each node
        process = subprocess.Popen(
            [
                "python3",
                "node2.py",
                str(node_id),
                str(num_nodes),
                str(f),
                genesis_block_json,
                str(block_interval),
                str(base_port),
                node_peers_json,
                node_public_keys_json,  # Pass public keys to each node
                node_private_key  # Pass the private key to the node
            ]
        )
        processes.append(process)

    # Keep the main process running and handle graceful shutdown
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down nodes...")
        for p in processes:
            p.terminate()
        for p in processes:
            p.wait()
        print("All nodes have been shut down.")
if __name__ == "__main__":
    simulate_voting_process()
