import socket
import threading
import json
import time
import random
import queue
import uuid
import os
import multiprocessing
import sys
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.exceptions import InvalidSignature
import base64
from concurrent.futures import ThreadPoolExecutor
from bitarray import bitarray
import math
import hashlib
class CustomBloomFilter:
    def __init__(self, capacity, error_rate):
        self.capacity = capacity
        self.error_rate = error_rate
        self.num_bits = math.ceil(-(capacity * math.log(error_rate)) / (math.log(2) ** 2))
        self.num_hashes = math.ceil((self.num_bits / capacity) * math.log(2))
        self.bitarray = bitarray(self.num_bits)
        self.bitarray.setall(0)

    def add(self, item):
        for i in range(self.num_hashes):
            digest = self.hash_item(item, i)
            self.bitarray[digest % self.num_bits] = True

    def __contains__(self, item):
        for i in range(self.num_hashes):
            digest = self.hash_item(item, i)
            if not self.bitarray[digest % self.num_bits]:
                return False
        return True

    def hash_item(self, item, seed):
        # Use hashlib to create a consistent hash based on the seed and the item
        return int(hashlib.sha256((str(seed) + item).encode('utf-8')).hexdigest(), 16)

    def serialize(self):
        """
        Serialize the bloom filter's bit array and other relevant attributes.
        """
        return json.dumps({
            'bitarray': base64.b64encode(self.bitarray.tobytes()).decode('utf-8'),
            'capacity': self.capacity,
            'error_rate': self.error_rate,
            'num_bits': self.num_bits,
            'num_hashes': self.num_hashes
        })

    @staticmethod
    def deserialize(data):
        """
        Deserialize the bloom filter from the serialized data.
        """
        obj = json.loads(data)
        bloom_filter = CustomBloomFilter(capacity=obj['capacity'], error_rate=obj['error_rate'])
        bloom_filter.bitarray = bitarray()
        bloom_filter.bitarray.frombytes(base64.b64decode(obj['bitarray']))
        bloom_filter.num_bits = obj['num_bits']
        bloom_filter.num_hashes = obj['num_hashes']
        return bloom_filter

# Deserialize the private key from PEM (string format)
def deserialize_private_key(private_key_str):
    return serialization.load_pem_private_key(
        private_key_str.encode('utf-8'),  # Convert string to bytes
        password=None,  # No password used
        backend=default_backend()
    )

# Function to sign data using the private key
def sign_data(private_key, data):
    # Ensure data is bytes before signing
    if isinstance(data, str):
        data = data.encode()  # Convert to bytes if it's a string

    signature = private_key.sign(
        data,  # Data must be bytes
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode()  # Return the base64 encoded signature as a string

# Function to verify the signature using the public key
def verify_signature(public_key, signature, data):
    try:
        public_key.verify(
            base64.b64decode(signature),  # Decode the base64 encoded signature
            data,  # Data must be in bytes
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return True  # If the signature is valid, return True
    except InvalidSignature:
        return False  # If the signature is invalid, return False


# Serialization and Deserialization Functions

def generate_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    return private_key, public_key

# Serialize public and private keys for storage and transmission
def serialize_public_key(public_key):
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

def serialize_private_key(private_key):
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )


# Updated Block serialization function
def serialize_block(block):
    block_dict = {
        'block_hash': block.block_hash,
        'height': block.height,
        'previous_block_hash': block.previous_block_hash,
        'proposer_id': block.proposer_id,
        'signature': block.signature  # Include the signature in serialization
    }
    return json.dumps(block_dict)

# Updated Block deserialization function
def deserialize_block(block_json):
    data = json.loads(block_json)
    block = Block(
        #block_hash=data['block_hash'],
        height=data['height'],
        previous_block_hash=data['previous_block_hash'],
        proposer_id=data['proposer_id']
    )
    block.signature = data['signature']  # Restore the signature
    return block

# Updated Vote serialization function with Bloom Filter serialized
def serialize_vote(vote):
    vote_dict = {
        'height': vote.height,
        'block_hash': vote.block_hash,
        'voter': vote.voter,
        'signature': vote.signature,
        'received_votes_bitstring': vote.received_votes_bitstring,
        'bloom_filter': vote.bloom_filter.serialize()  # Serialize the bloom filter
    }
    return json.dumps(vote_dict)



# Updated Vote deserialization function with Bloom Filter restored from serialized data
def deserialize_vote(vote_json):
    data = json.loads(vote_json)
    
    vote = Vote(
        height=data['height'],
        block_hash=data['block_hash'],
        voter=data['voter'],
        bloom_filter= CustomBloomFilter.deserialize(data['bloom_filter']),  # Restore the bloom filter
        received_votes_bitstring=data['received_votes_bitstring'],
        N=len(data['received_votes_bitstring'])  # Set N based on the bitstring size
    )
    vote.signature = data['signature']
    return vote

# Block Class

class Block:
    def __init__(self, height, previous_block_hash=None, proposer_id=None):
        self.height = height
        self.previous_block_hash = previous_block_hash
        self.proposer_id = proposer_id
        self.signature = None  # Signature of the block
        self.block_hash = self.compute_hash()  # Compute the block's real hash

    def compute_hash(self):
        data = f"{self.height}:{self.previous_block_hash}:{self.proposer_id}".encode()
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(data)
        return base64.b64encode(digest.finalize()).decode()  # Return base64 encoded hash

    def sign_block(self, private_key):
        data_to_sign = f"{self.block_hash}:{self.height}:{self.previous_block_hash}:{self.proposer_id}"
        self.signature = sign_data(private_key, data_to_sign)

    def verify_block(self, public_key):
        data_to_verify = f"{self.block_hash}:{self.height}:{self.previous_block_hash}:{self.proposer_id}"
        return verify_signature(public_key, self.signature, data_to_verify)

# Vote Class

class Vote:
    def __init__(self, height, block_hash, voter, received_votes_bitstring, bloom_filter, N):
        self.height = height  # Height at which the vote is cast
        self.block_hash = block_hash  # The hash of the block this vote is for
        self.voter = voter  # The voter who cast this vote
        self.compliance = None
        # The following attributes will be reconstructed locally
        self.block = None  # The actual Block object, reconstructed locally
        self.previous_votes = []  # List of previous Vote objects, reconstructed locally
        self.G = None  # The vote graph, built locally
        self.VC = None  # The vote count, calculated locally
        self.signature = None  # Signature of the vote
        self.block_hash_map = {}  # Reference to the node's block hash map
        self.received_votes_bitstring = received_votes_bitstring # [0] * N  # Initialize an N-bit string (list of 0s)
        self.bloom_filter =bloom_filter #BloomFilter(capacity=N, error_rate=1e-7)  # Create a Bloom Filter for vote hashes
        self.vote_hash = self.compute_hash()  # Compute the real hash for the vote  should conpute again after adding bloom filter
        self.N = N  # The total number of nodes (determines bitstring size)

    def compute_hash(self):
        """
        Compute the hash using block details, received votes bitstring, and the serialized bloom filter.
        """
        # Prepare data for hashing, including bitstring and serialized bloom filter contents
        bitstring_str = ''.join(map(str, self.received_votes_bitstring))  # Convert the bitstring to a string
        bloom_filter_content = (self.bloom_filter.serialize()).encode('utf-8')  # Serialize the bloom filter
    
        # Concatenate all relevant parts for hashing
        data = f"{self.height}:{self.block_hash}:{self.voter}:{bitstring_str}".encode() + bloom_filter_content
    
        # Hash the concatenated string and bloom filter bytes
        digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
        digest.update(data)
        return base64.b64encode(digest.finalize()).decode()
        
    def sign_vote(self, private_key):
        """
        Sign the vote, including the hash, bitstring, and serialized bloom filter.
        """
        # Prepare data to sign based on vote hash, bitstring, and bloom filter
        bitstring_str = ''.join(map(str, self.received_votes_bitstring))
        
        # Serialize the bloom filter (which is already a string, no need to encode it)
        bloom_filter_content = self.bloom_filter.serialize()  # This is a string
        
        # Concatenate the string components first
        data_to_sign = f"{self.vote_hash}:{self.height}:{self.block_hash}:{self.voter}:{bitstring_str}{bloom_filter_content}"
        
        # Convert the entire string to bytes once, before signing
        data_to_sign_bytes = data_to_sign.encode()  # Encode the final string to bytes
        
        # Sign the data using the provided private key
        self.signature = sign_data(private_key, data_to_sign_bytes)

    
        
    def verify_vote(self, public_key):
        """
        Verify the vote signature by ensuring the data matches the signature, including serialized bloom filter info.
        """
        # Prepare data to verify based on vote hash, bitstring, and bloom filter
        bitstring_str = ''.join(map(str, self.received_votes_bitstring))  # Convert bitstring to string format
        bloom_filter_content = self.bloom_filter.serialize()  # This is a string, no need to encode again
        
        # Concatenate the components into a single string
        data_to_verify = f"{self.vote_hash}:{self.height}:{self.block_hash}:{self.voter}:{bitstring_str}{bloom_filter_content}"
        
        # Convert the entire string to bytes
        data_to_verify_bytes = data_to_verify.encode()  # Now encode the final string to bytes
    
        # Verify the signature using the public key
        return verify_signature(public_key, self.signature, data_to_verify_bytes)
    


    def get_vote_hash(self):
        return f"vote_{self.voter}_{self.height}_{self.vote_hash}"

    def __eq__(self, other):
        return isinstance(other, Vote) and self.get_vote_hash() == other.get_vote_hash()

    def __hash__(self):
        return hash(self.get_vote_hash())


    def update_received_votes(self, previous_votes):
        """
        Updates the received votes bitstring and adds votes to the bloom filter.
        """
        for vote in previous_votes:
            voter_id = vote.voter
            if voter_id < self.N:
                self.received_votes_bitstring[voter_id] = 1  # Mark this node as having received the vote
                self.bloom_filter.add(vote.get_vote_hash())  # Add the vote hash to the bloom filter

    def check_bloom_filter(self, vote_hash):
        """
        Check if the vote hash is present in the bloom filter.
        """
        return vote_hash in self.bloom_filter

    def build_vote_graph(self):
        """
        Build the vote graph G using reconstructed previous votes.
        """
        if self.G is not None:
            return self.G
        graph = {}
        queue_ = [self]
        processed_votes = set()

        while queue_:
            current_vote = queue_.pop(0)
            if current_vote in processed_votes:
                continue
            processed_votes.add(current_vote)
            graph[current_vote] = current_vote.previous_votes
            for prev_vote in current_vote.previous_votes:
                if prev_vote.height >= 0 and prev_vote not in processed_votes:
                    queue_.append(prev_vote)
        self.G = graph
        return graph

    def calculate_vote_count(self):
        """
        Calculate the aggregated vote count (VC) for this vote, filtering out double voting.
        """
        if self.VC is not None:
            return self.VC

        vote_count = {}  # Start with an empty vote count for each block
        voters_by_height = {}  # Track voters by height
        double_voters = set()  # Track voters who have double-voted

        # Organize votes by height for easier processing
        height_to_votes = self.organize_votes_by_height()

        # Process votes from height 1 upwards
        for height in sorted(height_to_votes.keys()):
            if height == self.height:  # should not count itself.
                continue
            for vote in height_to_votes[height]:
                voter = vote.voter

                # Detect and handle double-voting
                if voter not in double_voters:
                    if voter in voters_by_height.get(height, set()):
                        # Double voting detected
                        # Mark the voter as a double voter
                        double_voters.add(voter)
                        # Remove any previously counted votes by this voter
                        previous_vote = [v for v in height_to_votes[height] if v.voter == voter][0]
                        self.remove_vote_from_chain(previous_vote, vote_count)
                        # Ignore this vote and any future votes by this voter at this height
                    else:
                        # First valid vote from this voter at this height
                        voters_by_height.setdefault(height, set()).add(voter)
                        self.add_vote_to_chain_iterative(vote, vote_count)  # Add the vote to the chain

        self.VC = vote_count
        return vote_count

    def organize_votes_by_height(self):
        """
        Organize the votes in the graph G by their height and collect the blocks associated with those votes.
        Returns a dictionary where the keys are heights and the values are lists of votes at that height.
        """
        height_to_votes = {}
        for vote in self.G:
            if vote.height not in height_to_votes:
                height_to_votes[vote.height] = []
            height_to_votes[vote.height].append(vote)
        return height_to_votes

    def add_vote_to_chain_iterative(self, vote, vote_count):
        """
        Iteratively add the vote to the block and all previous blocks in the chain.
        This ensures that votes are propagated upward through the chain.
        """
        current_block = vote.block

        # Propagate the vote count up through all ancestor blocks
        while current_block:
            if current_block.block_hash in vote_count:
                vote_count[current_block.block_hash] += 1
            else:
                vote_count[current_block.block_hash] = 1

            # Move to the previous block in the chain
            if current_block.previous_block_hash:
                current_block = self.block_hash_map.get(current_block.previous_block_hash)
            else:
                break

    def remove_vote_from_chain(self, vote, vote_count):
        """Remove the vote from the block and all previous blocks in the chain due to double voting."""
        current_block = vote.block

        while current_block:
            # Decrement the vote count for the current block
            if current_block.block_hash in vote_count and vote_count[current_block.block_hash] > 0:
                vote_count[current_block.block_hash] -= 1

            # Move to the previous block in the chain
            if current_block.previous_block_hash:
                current_block = self.block_hash_map.get(current_block.previous_block_hash)
            else:
                break

    def is_compliant_vote(self):
        """
        Iteratively determine if the current vote is compliant based on two conditions:
        1. All votes in its vote graph G must be compliant.
        2. The vote is for a block that stems from a chain of blocks where each block at each height
           has the highest vote count in its branch, according to the VC of the vote.
        """
        if self.compliance is not None:
            return self.compliance

        # Build the vote graph and calculate vote counts
        self.build_vote_graph()
        self.calculate_vote_count()

        # Step 1: Ensure all votes in its vote graph G are compliant
        queue_ = [self]  # Start with the current vote
        processed_votes = set()  # Track which votes have been processed

        while queue_:
            current_vote = queue_.pop(0)

            if current_vote in processed_votes:
                continue  # Skip already processed votes
            processed_votes.add(current_vote)

            # Add the previous votes of this vote to the queue
            queue_.extend(current_vote.previous_votes)

            # Check if the previous votes are compliant
            for prev_vote in current_vote.previous_votes:
                if not prev_vote.is_compliant_vote():
                    self.compliance = False
                    return self.compliance

        # Step 2: Check that the vote is for the block with the highest vote count at each height in the branch
        self.compliance = self.is_highest_vote_count_in_branch()
        return self.compliance

    def is_highest_vote_count_in_branch(self):
        """Check if the vote is for a block with the highest vote count at each height in the branch."""
        current_block = self.block

        while current_block.previous_block_hash:
            # Get the previous block in the chain
            previous_block = self.block_hash_map.get(current_block.previous_block_hash)
            if not previous_block:
                return False

            # Get all blocks at this height (siblings in the branch)
            sibling_blocks = self.get_blocks_at_height_by_votes(previous_block.height)
            sibling_blocks = [sibling_block for sibling_block in sibling_blocks if sibling_block.previous_block_hash==previous_block.previous_block_hash]

            if not sibling_blocks:
                return True  # No competition

            # Find the highest vote count at this height
            max_vote_count = max(self.VC.get(block.block_hash, 0) for block in sibling_blocks)

            # Get all blocks that have the highest vote count (accounting for ties)
            highest_blocks = [block for block in sibling_blocks if self.VC.get(block.block_hash, 0) == max_vote_count]

            # If the current block's previous block is in the set of highest blocks, the branch is valid
            if previous_block in highest_blocks:
                current_block = previous_block  # Proceed to the previous block in the chain
            else:
                return False

        return True

    def get_blocks_at_height_by_votes(self, height):
        """
        Retrieve all blocks at a given height by examining the votes.
        """
        blocks_at_height = set()
        for vote in self.G:
            if vote.height == height:
                blocks_at_height.add(vote.block)
        return list(blocks_at_height)

# Node Class

class Node:
    def __init__(self, node_id, num_nodes, f, genesis_block, delta_bi, host='localhost', base_port=8000):
        self.node_id = node_id
        self.num_nodes = num_nodes
        self.f = f
        self.delta_bi = delta_bi
        self.blockchain = {0: [genesis_block]}
        self.block_hash_map = {genesis_block.block_hash: genesis_block}
        self.vote_hash_map = {}
        self.received_votes = []
        self.N_f_votes_required = num_nodes - f
        self.genesis_block = genesis_block
        self.most_recent_accepted = genesis_block.block_hash
        self.latest_vote_height = -1
        self.vote_height = -1
        self.peers = {}
        self.host = host
        self.port = base_port + node_id
        self.server_socket = None
        self.stop_event = threading.Event()
        self.message_queue = queue.Queue()
        self.blocklock = threading.Lock()
        self.votelock = threading.Lock()
        self.pending_votes = []
        self.executor = ThreadPoolExecutor(max_workers=30)  # Adjust the number as needed
        # The node's private key for signing
        self.private_key = None  # Set after initialization

        self.vote_hash_map_by_node = {}  # Track received vote hashes by node and height
        
        # Public keys of all nodes
        self.public_keys = {}

    def set_public_keys(self, public_keys_json):
        self.public_keys = {
            int(node_id): serialization.load_pem_public_key(public_key.encode(), backend=default_backend())
            for node_id, public_key in public_keys_json.items()
        }

        # Public keys of other nodes (to be shared during node initialization)
        self.start_server()
        threading.Thread(target=self.process_pending_votes_thread, daemon=True).start()


    def start_working(self):
        threading.Thread(target=self.voting, daemon=True).start()
        threading.Thread(target=self.proposing_block, daemon=True).start()
        threading.Thread(target=self.check_consensus_thread, daemon=True).start()

    def start_server(self):
        """Start the server to listen for incoming connections."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.settimeout(1.0)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        threading.Thread(target=self.accept_connections, daemon=True).start()
        threading.Thread(target=self.process_message_queue, daemon=True).start()

    def accept_connections(self):
        """Accept incoming connections from peers."""
        while not self.stop_event.is_set():
            try:
                conn, addr = self.server_socket.accept()
                threading.Thread(target=self.handle_connection, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[{time.time()}] Node {self.node_id} accept_connections exception: {e}\n")
                break

    def handle_connection(self, conn):
        """Handle incoming messages from a peer."""
        with conn:
            data = b''
            try:
                while not self.stop_event.is_set():
                    temp_ = conn.recv(4096)
                    if not temp_:
                        break
                    data += temp_
                message = json.loads(data.decode())
                data = b''
                self.message_queue.put(message)
            except Exception as e:
                pass

    def process_message_queue(self):
        while not self.stop_event.is_set():
            if not self.message_queue.empty():
                message = self.message_queue.get()
                # Submit the task to the thread pool
                self.executor.submit(self.process_message, message)
            else:
                time.sleep(0.001)

    def send_message(self, node_id, message_type, content):
        """Send a message to another node."""
        if node_id not in self.peers:
            return
        host, port = self.peers[node_id]
        message = {
            'type': message_type,
            'content': content,
            'sender_id': self.node_id
        }
        serialized_message = json.dumps(message).encode()
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                s.sendall(serialized_message)
        except Exception as e:
            pass

    def process_message(self, message):
        """Process a received message."""
        message_type = message['type']
        content = message['content']
        sender_id = message['sender_id']

        if message_type == 'block':
            block_json = content
            block = deserialize_block(block_json)
            self.receive_block(block)
        elif message_type == 'vote':
            vote_json = content
            vote = deserialize_vote(vote_json)
            self.receive_vote(vote)
        elif message_type == 'request_block':
            block_hash = content['block_hash']
            self.handle_block_request(sender_id, block_hash)
        elif message_type == 'request_vote':
            vote_voter = content['voter']
            vote_height = content['vote_height']
            print(f"handling_request: voter{vote_voter}, vote_height{vote_height}")
            self.handle_vote_request(sender_id, vote_voter,vote_height)
        else:
            print(f"[{time.time()}] Node {self.node_id} received unknown message type: {message_type}\n")

    def receive_block(self, block):
        """Receive a block proposed by another node."""
        with self.blocklock:
            if block.block_hash not in self.block_hash_map:
                self.block_hash_map[block.block_hash] = block
                self.blockchain.setdefault(block.height, []).append(block)
                #print (f"{self.node_id} received  a block {block.block_hash}")
            else:
                pass  # Block already received

    def receive_vote(self, vote):
        """Receive a vote from another node and process it."""
        with self.votelock:
            # Reconstruct the vote's block and previous votes
            if self.reconstruct_vote_data(vote):
                if vote not in self.received_votes:
                    voter_public_key = self.public_keys.get(vote.voter)
                    if voter_public_key and vote.verify_vote(voter_public_key):
                        # Process vote
                        if vote.is_compliant_vote():
                            self.store_vote(vote)
                            self.received_votes.append(vote)
                            #print(f"Node {self.node_id} received a compliant vote from node {vote.voter}, which is for a block of {vote.block.height}")
                            self.latest_vote_height = max(self.latest_vote_height, vote.height)
                        else:
                            print(f"Node {self.node_id} received a non-compliant vote from node {vote.voter}")
                            pass  # Non-compliant vote
                    else:
                        print(f"Node {self.node_id} received an invalid vote from node {vote.voter}")
            else:
                pass  # Missing data, vote added to pending_votes

    def reconstruct_vote_data(self, vote):
        """Reconstruct the block and previous votes referenced by the vote."""
        missing_data = False

        # Reconstruct the block
        if vote.block_hash in self.block_hash_map:
            vote.block = self.block_hash_map[vote.block_hash]
        else:
            # Request the block from peers
            self.send_message(vote.voter, 'request_block', {'block_hash': vote.block_hash})
            #self.request_block(vote.block_hash)
            missing_data = True

        # Reconstruct previous votes
        vote.previous_votes= []
        for i in range(len(vote.received_votes_bitstring)):
            if vote.received_votes_bitstring[i] == 1:
                # This indicates that the vote from node i was received
                vote_hash = self.get_vote_hash_from_bloom_filter(vote, i)
               # print ("votehash is", vote_hash, vote.received_votes_bitstring)
                if vote_hash:
                    prev_vote = self.vote_hash_map.get(vote_hash)
                    vote.previous_votes.append(prev_vote)
                else:
                    self.send_message(vote.voter, 'request_vote', {'voter':i, 'vote_height': vote.height-1})
                    missing_data = True
            
       # for prev_vote_hash in vote.previous_vote_hashes:
       #     prev_vote = self.vote_hash_map.get(prev_vote_hash)
       #     if prev_vote:
       #         vote.previous_votes.append(prev_vote)
       #     else:
       #         # Request the previous vote from peers
       #         self.send_message(vote.voter, 'request_vote', {'vote_hash': prev_vote_hash})
       #         #self.request_vote(prev_vote_hash)
       #         missing_data = True

        if missing_data:
            # Store the vote to be processed later
            if vote not in self.pending_votes:
                self.pending_votes.append(vote)
        else:
            # Set block hash map reference
            vote.vote_hash = vote.compute_hash()
            vote.block_hash_map = self.block_hash_map
        return not missing_data

    def process_pending_votes_thread(self):
        """Background thread to process pending votes."""
        while not self.stop_event.is_set():
            self.process_pending_votes()

    def process_pending_votes(self):
        """Process votes that were pending due to missing data."""
        for vote in list(self.pending_votes):
            can_process = True
            # Check if all required data is available
            if vote.block_hash not in self.block_hash_map:
                can_process = False
                if random.randint(0, 1000) == 3:
                    self.request_block(vote.block_hash)
                #self.request_block(vote.block_hash)

            for i in range(len(vote.received_votes_bitstring)):
                if vote.received_votes_bitstring[i] == 1:
                    # This indicates that the vote from node i was received
                    vote_hash = self.get_vote_hash_from_bloom_filter(vote, i)
                    if not vote_hash:
                        can_process = False
                        if random.randint(0, 1000) == 5:
                            for peer_id in self.peers:
                                self.send_message(peer_id, 'request_vote', {'voter':i, 'vote_height': vote.height-1})
            if can_process:
                self.pending_votes.remove(vote)
                self.receive_vote(vote)  # Now we can process the vote

    def request_block(self, block_hash):
        """Broadcast a block request to all peers."""
        for peer_id in self.peers:
            self.send_message(peer_id, 'request_block', {'block_hash': block_hash})

    def handle_block_request(self, requester_id, block_hash):
        """Send the requested block to the requester."""
        block = self.block_hash_map.get(block_hash)
        if block:
            block_json = serialize_block(block)
            self.send_message(requester_id, 'block', block_json)

    def handle_vote_request(self, requester_id, vote_voter, vote_height):
        """Send the requested vote to the requester."""
        my_vote = [vote for vote in self.received_votes if vote.height == vote_height+1 and vote.voter==self.node_id] # find the vote of its own
        if len(my_vote)>0:
            for vote in my_vote[0].previous_votes:
                if vote.height == vote_height and vote.voter == vote_voter:
                    vote_json = serialize_vote(vote)
                    self.send_message(requester_id, 'vote', vote_json)

    def store_vote(self, vote):
        """
        Store the vote and update the vote hash map by node.
        """
        self.vote_hash_map[vote.get_vote_hash()] = vote
        if vote.voter not in self.vote_hash_map_by_node:
            self.vote_hash_map_by_node[vote.voter] = {}
        if vote.height not in self.vote_hash_map_by_node[vote.voter]:
            self.vote_hash_map_by_node[vote.voter][vote.height] = [vote.get_vote_hash()]
        else:
            self.vote_hash_map_by_node[vote.voter][vote.height].append (vote.get_vote_hash())

    def get_vote_hash_from_bloom_filter(self, vote, node_id):
        """
        Retrieve the vote hash from the bloom filter using the node ID.
        This assumes we have a map that tracks which votes each node has cast.
        """
        possible_vote_hashes = self.vote_hash_map_by_node.get(node_id, []).get(vote.height-1)
        #print ("potential:", possible_vote_hashes)
        for possible_vote_hash in possible_vote_hashes:
            #print ("potential:", possible_vote_hash, vote.check_bloom_filter(possible_vote_hash))
            if possible_vote_hash and vote.check_bloom_filter(possible_vote_hash):
                return possible_vote_hash
        return None

    def create_block(self, previous_block):
        return Block(
            #block_hash=f"block_{self.node_id}_{previous_block.height + 1}_{random.randint(0, 9999)}",
            height=previous_block.height + 1,
            previous_block_hash=previous_block.block_hash,
            proposer_id=self.node_id
        )

    def propose_block(self):
        """Propose a block at the current time based on the shared clock."""
        current_time = int(time.time())
        proposer_index = (current_time // self.delta_bi) % self.num_nodes

        if proposer_index == self.node_id:
            latest_block_candidate_height = self.largest_height_with_compliant_votes(self.received_votes, self.num_nodes, self.f)
            if latest_block_candidate_height != -1:
                branches = self.get_most_supported_blocks(0, latest_block_candidate_height)
                if branches:
                    latest_block_candidate_branch = random.choice(branches)
                    latest_block_candidate = self.find_leaf_block(latest_block_candidate_branch, self.blockchain)
                else:
                    latest_block_candidate = self.genesis_block
            else:
                latest_block_candidate = self.genesis_block
            if latest_block_candidate.height + 1 in self.blockchain:
                for other_block in self.blockchain[latest_block_candidate.height + 1]:
                    if other_block.previous_block_hash == latest_block_candidate.block_hash:
                        return None  # Someone already proposed a block of the same height at this branch.
            new_block = self.create_block(latest_block_candidate)
            #print(f"Private key:{self.private_key}")
            new_block.sign_block(self.private_key)
            self.blockchain.setdefault(new_block.height, []).append(new_block)
            self.block_hash_map[new_block.block_hash] = new_block
            print(f"[{time.time()}] Node {self.node_id} proposed block {new_block.block_hash} at consensus height {new_block.height} extending from {latest_block_candidate.block_hash}\n")
            # Broadcast the block to all peers
            block_json = serialize_block(new_block)
            for peer_id in self.peers:
                self.send_message(peer_id, 'block', block_json)
            return new_block  # Return block to broadcast to other nodes
        return None

    def voting(self):
        while not self.stop_event.is_set():
            self.cast_vote()
            #time.sleep(0.5)

    def proposing_block(self):
        last_done = 0
        while not self.stop_event.is_set():
            te = time.time() // self.delta_bi
            if te % self.num_nodes == self.node_id and last_done != te:
                self.propose_block()
                last_done = te

    def cast_vote(self):
        """
        Cast a vote at the given vote height based on N-f compliant votes from previous heights.
        This function now uses the bloom filter and N-bit string to encapsulate previous votes.
        """
        vote_height = self.vote_height + 1
    
        if vote_height == 0:
            # At height 0, vote for the Genesis block
            genesis_vote = Vote(0, self.genesis_block.block_hash, self.node_id, [0] * self.num_nodes , CustomBloomFilter(capacity=self.num_nodes, error_rate=1e-7), self.num_nodes)
            genesis_vote.block = self.genesis_block
            genesis_vote.block_hash_map = self.block_hash_map
            self.vote_height = vote_height
            # Broadcast the vote
            genesis_vote.vote_hash = genesis_vote.compute_hash()
            genesis_vote.sign_vote(self.private_key)  # Sign the vote
            vote_json = serialize_vote(genesis_vote)
            for peer_id in self.peers:
                self.send_message(peer_id, 'vote', vote_json)
            genesis_vote.build_vote_graph()
            genesis_vote.calculate_vote_count()
            self.store_vote(genesis_vote)
            self.received_votes.append(genesis_vote)
            print(f"[{time.time()}] Node {self.node_id} cast a vote for the Genesis block at vote height 0.\n")
            return genesis_vote  # Return vote to broadcast to other nodes
    
        # Find the block to vote for based on previous compliant votes
        block_to_vote_for = self.find_block_to_vote_for(vote_height)
        if block_to_vote_for is not None:
            previous_vote_height = vote_height - 1
            compliant_votes = [vote for vote in self.received_votes if vote.height == previous_vote_height and vote.is_compliant_vote()]
    
            if len(compliant_votes) >= self.N_f_votes_required:
                # Create the list of previous vote hashes and update the bitstring and bloom filter
                previous_vote_hashes = [vote.get_vote_hash() for vote in compliant_votes]
    
                # Create the new vote
                new_vote = Vote(vote_height, block_to_vote_for.block_hash, self.node_id, [0] *  self.num_nodes , CustomBloomFilter(capacity=self.num_nodes, error_rate=1e-7), self.num_nodes)
    
                # Update the N-bit string and bloom filter with previous votes
                new_vote.update_received_votes(compliant_votes)    
                new_vote.block = block_to_vote_for
                new_vote.previous_votes = compliant_votes
                new_vote.block_hash_map = self.block_hash_map
                new_vote.vote_hash = new_vote.compute_hash()
                # Sign the vote, including the bloom filter and bitstring
                new_vote.sign_vote(self.private_key)
    
                # Serialize the vote and broadcast it
                vote_json = serialize_vote(new_vote)
                for peer_id in self.peers:
                    self.send_message(peer_id, 'vote', vote_json)
    
                # Store and append the vote locally
                self.store_vote(new_vote)
                self.received_votes.append(new_vote)
                self.vote_height = vote_height
    
                # Build the vote graph and calculate vote count
                new_vote.build_vote_graph()
                new_vote.calculate_vote_count()
    
                print(f"[{time.time()}] Node {self.node_id} cast a vote for block {block_to_vote_for.block_hash} at vote height {vote_height}. "
                      f"This vote is linked to the following nodes' vote at vote height {vote_height - 1}: "
                      f"{[item.voter for item in new_vote.previous_votes]} The VC of this vote is: {new_vote.VC}\n")
    
                return new_vote  # Return the new vote to broadcast to other nodes
            else:
                print(f"[{time.time()}] Node {self.node_id} is waiting for more compliant votes at vote height {previous_vote_height}.\n")
        else:
            pass
           # print(f"[{time.time()}] Node {self.node_id} cannot find a block to vote at {vote_height}.\n")
        return None

    def find_block_to_vote_for(self, vote_height):
        """
        Find the block at vote height k that stems from the most supported blocks at each previous height.
        Recursively go back through previous heights to find the most supported branch.
        """
        if vote_height == 0:
            return self.genesis_block
        current_blocks = self.get_most_supported_blocks(0, vote_height - 1)
        if not current_blocks:
            return None
        for current_block in current_blocks:
            candidate_blocks = [block for block in self.blockchain.get(vote_height, []) if block.previous_block_hash == current_block.block_hash]
            if candidate_blocks:
                return candidate_blocks[0]
        return None

    def get_most_supported_blocks(self, starting_height, consensus_height):
        """
        Determine one block at the given consensus height that is part of the most supported branch.
        If there are multiple branches with the same vote count, randomly select one, provided that these branches have blocks at consensus_height.
        """
        # Step 1: Filter compliant votes at the given consensus_height
        compliant_votes = [vote for vote in self.received_votes if vote.block.height == consensus_height and vote.is_compliant_vote()]
        if len(compliant_votes) < self.N_f_votes_required:
            #print (f"The most supported block cannot be found because < requried number,{len(compliant_votes)} at height {consensus_height}")
          #  for vote in self.received_votes:
           #     print (self.node_id, "height:", vote.block.height)
            return []

        # Step 2: Create the virtual vote using compliant votes
        virtual_vote = self.create_virtual_vote(compliant_votes)

        # Step 3: Track both most and second most supported blocks at each level
        most_supported_blocks_at_each_level = {}
        highest_vote_count_at_height = {}

        # Step 4: Find the blocks with the most support at starting_height
        for block_hash, vote_count in virtual_vote.VC.items():
            block = self.block_hash_map.get(block_hash)
            if block.height == starting_height:
                if block.height not in highest_vote_count_at_height or vote_count > highest_vote_count_at_height[block.height]:
                    highest_vote_count_at_height[block.height] = vote_count
                    most_supported_blocks_at_each_level[block.height] = [block]
                elif vote_count == highest_vote_count_at_height[block.height]:
                    most_supported_blocks_at_each_level[block.height].append(block)

        if starting_height == consensus_height:
            most_supported_blocks = most_supported_blocks_at_each_level.get(starting_height, [])
            return [random.choice(most_supported_blocks)] if most_supported_blocks else []

        # Step 5: Trace down the most supported branches by checking ancestors at each height
        for h in range(starting_height + 1, consensus_height + 1):
            most_supported_blocks_at_each_level[h] = []
            highest_vote_count_at_height[h] = 0

            # For each block in the previous level, find its valid children
            for parent_block in most_supported_blocks_at_each_level[h - 1]:
                for block_hash, vote_count in virtual_vote.VC.items():
                    block = self.block_hash_map.get(block_hash)
                    if block and block.height == h and block.previous_block_hash == parent_block.block_hash:
                        if vote_count > highest_vote_count_at_height[h]:
                            highest_vote_count_at_height[h] = vote_count
                            most_supported_blocks_at_each_level[h] = [block]
                        elif vote_count == highest_vote_count_at_height[h]:
                            most_supported_blocks_at_each_level[h].append(block)

            # If no valid blocks are found at this level, the branch is incomplete
            if not most_supported_blocks_at_each_level[h]:
                return []

        # Step 6: Select one block from the most supported branch at consensus_height randomly
        most_supported_blocks = most_supported_blocks_at_each_level.get(consensus_height, [])
        return [random.choice(most_supported_blocks)] if most_supported_blocks else []

    def get_top_two_supported_branches(self, starting_height, consensus_height, G):
        """
        Find the most supported and second most supported branches starting from a given starting height.
        The most supported branch has the highest vote count at the starting height, and at each subsequent height,
        the ancestors in the chain also need to have the highest vote count among their siblings. The second-most branch
        has the second-highest vote count.
        """
        # Step 1: Filter compliant votes at the given starting height
        compliant_votes = [vote for vote in G if vote.block.height == consensus_height and vote.is_compliant_vote()]
        if len(compliant_votes) < self.N_f_votes_required:
            return None, None, None, None

        # Step 2: Create the virtual vote using compliant votes
        virtual_vote = self.create_virtual_vote(compliant_votes)

        # Step 3: Sort blocks in VC by vote count for the given starting height
        block_support_at_starting_height = []
        for block_hash, vote_count in virtual_vote.VC.items():
            block = self.get_block_by_hash(block_hash)
            if block and block.height == starting_height:
                block_support_at_starting_height.append((block_hash, vote_count))

        # Sort the blocks by vote count at the starting height in descending order
        block_support_at_starting_height.sort(key=lambda x: x[1], reverse=True)
        if len(block_support_at_starting_height) == 1:
            return block_support_at_starting_height[0][0], None, block_support_at_starting_height[0][1], 0

        if len(block_support_at_starting_height) == 0:
            return None, 0, 0, 0
        return block_support_at_starting_height[0][0], block_support_at_starting_height[1][0], block_support_at_starting_height[0][1], block_support_at_starting_height[1][1]

    def get_block_by_hash(self, block_hash):
        return self.block_hash_map.get(block_hash, None)

    def missing_votes_count(self, height, G):
        total_votes = self.num_nodes * (height)
        received_votes = sum(1 for vote in G if vote.height <= height)
        return total_votes - received_votes

    def largest_height_with_compliant_votes(self, received_votes, N, f):
        """
        Find the largest height where at least N-f compliant votes have been received.
        """
        N_f_threshold = N - f  # The threshold for compliant votes

        # Organize votes by height
        votes_by_height = {}
        for vote in received_votes:
            if vote.is_compliant_vote():  # Consider only compliant votes
                if vote.height not in votes_by_height:
                    votes_by_height[vote.height] = []
                votes_by_height[vote.height].append(vote)

        # Find the largest height with more than N-f compliant votes
        largest_compliant_height = -1  # Default value if no such height exists
        for height, votes in votes_by_height.items():
            if len(votes) >= N_f_threshold:
                largest_compliant_height = max(largest_compliant_height, height)

        return largest_compliant_height

    def find_leaf_block(self, root_block, blockchain):
        """
        Given the root block of a branch, find the leaf block (the one with no further extensions).
        If there are multiple leaf blocks, return the one with the largest height.
        """
        current_block = root_block
        max_height_block = current_block  # Track the block with the largest height

        # Traverse the chain starting from the root_block
        while True:
            next_blocks = [block for block in blockchain.get(current_block.height + 1, []) if block.previous_block_hash == current_block.block_hash]

            if not next_blocks:
                # If no blocks extend from the current block, it is a leaf block
                break

            # Choose the block with the largest height
            current_block = max(next_blocks, key=lambda block: block.height)
            max_height_block = current_block  # Update the max height block

        return max_height_block

    def create_virtual_vote(self, compliant_votes):
        virtual_vote = Vote(height=compliant_votes[0].height + 1, block_hash=None, voter=self.node_id,received_votes_bitstring=[0] * self.num_nodes ,bloom_filter=CustomBloomFilter(capacity=self.num_nodes, error_rate=1e-7), N=self.num_nodes)
        virtual_vote.previous_votes = compliant_votes
        virtual_vote.block_hash_map = self.block_hash_map  # Provide access to block hash map
        virtual_vote.build_vote_graph()
        virtual_vote.calculate_vote_count()
        return virtual_vote

    def is_consensus_reached(self, k_prime, k):
        """
        Determine if consensus is reached on a block at height k_prime based on votes at height k.
        """
        while k > k_prime + 1:
            compliant_votes = [vote for vote in self.received_votes if vote.block.height == k and vote.is_compliant_vote()]
            if len(compliant_votes) < self.N_f_votes_required:
                k -= 1
                continue
            ccount_=0
            for item in compliant_votes:
                most_supported_block, second_most_supported_block, v1, v2 = self.get_top_two_supported_branches(k_prime, k - 1, item.G)
                if not most_supported_block:
                    continue

                missing_votes_at_k_prime = self.missing_votes_count(k_prime, item.G)
                missing_votes_at_k_1 = self.missing_votes_count(k - 1, item.G)
                potential_double_votes = (k - k_prime) * self.f
                if v1 - potential_double_votes > v2 + missing_votes_at_k_1 - missing_votes_at_k_prime - 1:
                    ccount_+=1
                    #print(f"[{time.time()}] Node {self.node_id} thinks Consensus reached on block {most_supported_block} at height {k} because {most_supported_block} got {v1} supports, {second_most_supported_block} got {v2} supports, the missing votes are {missing_votes_at_k_1 - missing_votes_at_k_prime}.\n")
            if ccount_>=self.N_f_votes_required:
                self.most_recent_accepted = most_supported_block
                return most_supported_block
            k -= 1
        return False

    def check_consensus_thread(self):
        k_prime = 0
        first_time = {}
        while not self.stop_event.is_set():
            if self.latest_vote_height < k_prime:
                time.sleep(5)
                continue
            consensus = self.is_consensus_reached(k_prime, self.latest_vote_height)
            if consensus and consensus not in first_time:
                print(f"[{time.time()}] Node {self.node_id} thinks Consensus reached on block {consensus} at height {self.latest_vote_height} because {consensus} got enough supports.\n")
                k_prime += 1
                first_time[consensus] = True
            time.sleep(0.01)

    def stop(self):
        self.stop_event.set()
        if self.server_socket:
            self.server_socket.close()

# Function to run a node in a separate process

def run_node(node_id, num_nodes, f, genesis_block_json, delta_bi, base_port, peers, public_keys_json, private_key_pem):
    # Deserialize the genesis block
    genesis_block = deserialize_block(genesis_block_json)
    
    # Deserialize the private key
    private_key = serialization.load_pem_private_key(private_key_pem.encode(), password=None, backend=default_backend())

    # Create the node instance
    node = Node(node_id=node_id, num_nodes=num_nodes, f=f, genesis_block=genesis_block, delta_bi=delta_bi, base_port=base_port)
    
    # Set up peers and public keys
    node.peers = peers
    node.set_public_keys(public_keys_json)
    
    # Set the private key for signing
    node.private_key = private_key
    
    # Start the node
    node.start_working()

    # Keep the node running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        node.stop()
# Function to run a node in a separate process

# Main logic for node.py
if __name__ == "__main__":
    node_id = int(sys.argv[1])
    num_nodes = int(sys.argv[2])
    f = int(sys.argv[3])
    genesis_block_json = sys.argv[4]
    delta_bi = int(sys.argv[5])
    base_port = int(sys.argv[6])
    peers_json = sys.argv[7]
    public_keys_json = sys.argv[8]
    private_key_pem = sys.argv[9]  # Add the private key as a command-line argument

    peers = json.loads(peers_json)
    public_keys = json.loads(public_keys_json)

    run_node(node_id, num_nodes, f, genesis_block_json, delta_bi, base_port, peers, public_keys, private_key_pem)