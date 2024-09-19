import socket
import threading
import json
import time
import random
import queue
import uuid
import os
# Serialization and Deserialization Functions

def serialize_block(block):
    block_data = {
        'block_hash': block.block_hash,
        'height': block.height,
        'previous_block_hash': block.previous_block_hash,
        'proposer_id': block.proposer_id
    }
    return json.dumps(block_data)

def deserialize_block(block_json):
    block_data = json.loads(block_json)
    return Block(
        block_hash=block_data['block_hash'],
        height=block_data['height'],
        previous_block_hash=block_data['previous_block_hash'],
        proposer_id=block_data['proposer_id']
    )

def serialize_vote(vote):
    vote_data = {
        'height': vote.height,
        'block_hash': vote.block_hash,
        'voter': vote.voter,
        'previous_vote_hashes': vote.previous_vote_hashes,
    }
    return json.dumps(vote_data)

def deserialize_vote(vote_json):
    vote_data = json.loads(vote_json)
    vote = Vote(
        height=vote_data['height'],
        block_hash=vote_data['block_hash'],
        voter=vote_data['voter'],
        previous_vote_hashes=vote_data['previous_vote_hashes']
    )
    return vote

# Block Class

class Block:
    def __init__(self, block_hash, height, previous_block_hash=None, proposer_id=None):
        self.block_hash = block_hash  # Unique identifier for the block
        self.height = height  # Height of the block in the chain
        self.previous_block_hash = previous_block_hash  # Hash of the previous block
        self.proposer_id = proposer_id  # ID of the node that proposed this block

    def __repr__(self):
        return f"Block(hash={self.block_hash}, height={self.height}, proposer_id={self.proposer_id})"

    def __eq__(self, other):
        return isinstance(other, Block) and self.block_hash == other.block_hash

    def __hash__(self):
        return hash(self.block_hash)

# Vote Class

class Vote:
    def __init__(self, height, block_hash, voter, previous_vote_hashes):
        self.height = height  # Height at which the vote is cast
        self.block_hash = block_hash  # The hash of the block this vote is for
        self.voter = voter  # The voter who cast this vote
        self.previous_vote_hashes = previous_vote_hashes  # List of previous vote hashes
        self.compliance = None
        # The following attributes will be reconstructed locally
        self.block = None  # The actual Block object, reconstructed locally
        self.previous_votes = []  # List of previous Vote objects, reconstructed locally
        self.G = None  # The vote graph, built locally
        self.VC = None  # The vote count, calculated locally
        self.block_hash_map = {}  # Reference to the node's block hash map

    def get_vote_hash(self):
        return f"vote_{self.voter}_{self.height}"

    def __eq__(self, other):
        return isinstance(other, Vote) and self.get_vote_hash() == other.get_vote_hash()

    def __hash__(self):
        return hash(self.get_vote_hash())

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
            if height ==self.height: # should not count itself.
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
    def __init__(self, node_id, num_nodes, f, genesis_block, delta_bi, host='localhost', base_port=8000, retransmission_interval=1.0, max_retransmissions=5, message_loss_probability=0.1):
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
        self.vote_height =-1
        self.peers = {}
        self.host = host
        self.port = base_port + node_id
        self.server_socket = None
        self.stop_event = threading.Event()
        self.message_queue = queue.Queue()
        self.lock = threading.Lock()
        self.outgoing_messages = {}  # For retransmissions
        self.pending_votes = []      # Votes pending due to missing data
        self.retransmission_interval = retransmission_interval
        self.max_retransmissions = max_retransmissions
        self.message_loss_probability = message_loss_probability
        self.request_list={}
        self.start_server()
        threading.Thread(target=self.retransmission_handler, daemon=True).start()
        threading.Thread(target=self.process_pending_votes_thread, daemon=True).start()
    def start_working(self):
        threading.Thread(target=self.voting, daemon=True).start()
        threading.Thread(target=self.proposing_block, daemon=True).start()

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
                print(f"[{time.time()}]",f"Node {self.node_id} accept_connections exception: {e}")
                break

    def handle_connection(self, conn):
        """Handle incoming messages from a peer."""
        with conn:
            try:
                data = conn.recv(4096)
                while data:
                    messages = data.decode().split('\n')
                    for msg in messages:
                        if msg:
                            message = json.loads(msg)
                            self.message_queue.put(message)
                    data = conn.recv(4096)
            except Exception as e:
                pass

    def process_message_queue(self):
        """Process messages from the message queue."""
        while not self.stop_event.is_set():
            if not self.message_queue.empty():
                message = self.message_queue.get()
                self.process_message(message)
            else:
                time.sleep(0.001)

    def send_message(self, node_id, message_type, content):
        """Send a message to another node with retransmission logic."""
        if node_id not in self.peers:
            return
        host, port = self.peers[node_id]
        message_id = str(uuid.uuid4())  # Unique message ID
        message = {
            'type': message_type,
            'content': content,
            'sender_id': self.node_id,
            'message_id': message_id
        }
        serialized_message = json.dumps(message) + '\n'
        self.outgoing_messages[message_id] = {
            'message': serialized_message,
            'peer_id': node_id,
            'attempts': 0,
            'timestamp': time.time()
        }
        self.attempt_send_message(message_id)

    def attempt_send_message(self, message_id):
        """Attempt to send a message, handling retransmissions."""
        message_info = self.outgoing_messages.get(message_id)
        if not message_info:
            return
        peer_id = message_info['peer_id']
        host, port = self.peers[peer_id]
        try:
            # Simulate message loss
            if random.random() < self.message_loss_probability:
                # Message is lost; do not send
                return
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((host, port))
                s.sendall(message_info['message'].encode())
            message_info['attempts'] += 1
            message_info['timestamp'] = time.time()
        except Exception as e:
            # Message failed to send; will retry
            pass

    def retransmission_handler(self):
        """Background thread to handle message retransmissions."""
        while not self.stop_event.is_set():
            current_time = time.time()
            for message_id, message_info in list(self.outgoing_messages.items()):
                if current_time - message_info['timestamp'] > self.retransmission_interval:
                    if message_info['attempts'] < self.max_retransmissions:
                        self.attempt_send_message(message_id)
                    else:
                        # Exceeded max retransmissions; give up
                        try:
                            del self.outgoing_messages[message_id]
                        except:
                            pass
            time.sleep(0.1)

    def send_acknowledgment(self, node_id, ack_message):
        """Send an acknowledgment to a peer."""
        if node_id not in self.peers:
            return
        host, port = self.peers[node_id]
        serialized_ack = json.dumps(ack_message) + '\n'
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                s.connect((host, port))
                s.sendall(serialized_ack.encode())
        except Exception as e:
            pass

    def process_message(self, message):
        """Process a received message."""
        message_type = message['type']
        content = message['content']
        sender_id = message['sender_id']
        message_id = message.get('message_id')

        # Send acknowledgment
        if message_id:
            ack_message = {
                'type': 'ack',
                'content': {'message_id': message_id},
                'sender_id': self.node_id
            }
            self.send_acknowledgment(sender_id, ack_message)

        if message_type == 'ack':
            acked_message_id = content['message_id']
            if acked_message_id in self.outgoing_messages:
                del self.outgoing_messages[acked_message_id]
        elif message_type == 'block':
            block = deserialize_block(content)
            try:
                del self.request_list[str(sender_id)+'request_block'+str(block.block_hash)]
            except  Exception as e:
                pass
            self.receive_block(block)
        elif message_type == 'vote':
            vote = deserialize_vote(content)
            try:
                del self.request_list[str(sender_id)+'request_vote'+str(vote.get_vote_hash())]
            except Exception as e:
                pass
            self.receive_vote(vote)
        elif message_type == 'request_block':
            block_hash = content['block_hash']
            self.handle_block_request(sender_id, block_hash)
        elif message_type == 'request_vote':
            vote_hash = content['vote_hash']
            self.handle_vote_request(sender_id, vote_hash)
        else:
            print(f"[{time.time()}]",f"Node {self.node_id} received unknown message type: {message_type}")

    def receive_block(self, block):
        """Receive a block proposed by another node."""
        with self.lock:
            if block.block_hash not in self.block_hash_map:
                self.block_hash_map[block.block_hash] = block
                self.blockchain.setdefault(block.height, []).append(block)
                #print(f"[{time.time()}]",f"Node {self.node_id} received block {block.block_hash} at height {block.height} from node {block.proposer_id}.")
                # After receiving a block, check if any pending votes can be processed
            else:
                pass  # Block already received

    def receive_vote(self, vote):
        """Receive a vote from another node and process it."""
        with self.lock:
            # Reconstruct the vote's block and previous votes
            if self.reconstruct_vote_data(vote):
                if vote not in self.received_votes:
                    if vote.is_compliant_vote():
                        self.store_vote(vote)
                        self.received_votes.append(vote)
                        self.latest_vote_height = max(self.latest_vote_height, vote.height)
                        #print(f"[{time.time()}]",f"Node {self.node_id} received compliant vote for block {vote.block_hash} at height {vote.height} from node {vote.voter}.")
                        # After receiving a vote, check if any pending votes can be processed
                    else:
                        print(f"[{time.time()}]",f"Node {self.node_id} ignored non-compliant vote from node {vote.voter}.")

    def reconstruct_vote_data(self, vote):
        """Reconstruct the block and previous votes referenced by the vote."""
        missing_data = False

        # Reconstruct the block
        if vote.block_hash in self.block_hash_map:
            vote.block = self.block_hash_map[vote.block_hash]
        else:
            # Request the block from peers
            self.request_block(vote.block_hash,vote.voter)
            missing_data = True

        # Reconstruct previous votes
        k=[]
        for prev_vote_hash in vote.previous_vote_hashes:
            prev_vote = self.vote_hash_map.get(prev_vote_hash)
            if prev_vote:
                k.append(prev_vote)
            else:
                # Request the previous vote from peers
                self.request_vote(prev_vote_hash,vote.voter)
                k=[]
                missing_data = True

        if missing_data:
            # Store the vote to be processed later
            self.pending_votes.append(vote)
        else:
            # Set block hash map reference
            vote.block_hash_map = self.block_hash_map
            vote.previous_votes=k
        return not missing_data
        

    def process_pending_votes_thread(self):
        """Background thread to process pending votes."""
        while not self.stop_event.is_set():
            self.process_pending_votes()
            time.sleep(0.1)

    def process_pending_votes(self):
        """Process votes that were pending due to missing data."""
        for vote in list(self.pending_votes):
            can_process = True
            # Check if all required data is available
            if vote.block_hash not in self.block_hash_map:
                can_process = False
                self.request_block(vote.block_hash,vote.voter)
            for prev_vote_hash in vote.previous_vote_hashes:
                if prev_vote_hash not in self.vote_hash_map:
                    can_process = False
                    self.request_vote(prev_vote_hash,vote.voter)
            if can_process:
                self.pending_votes.remove(vote)
                self.receive_vote(vote)  # Now we can process the vote

    def request_block(self, block_hash,voter):
        """Request a block from peers."""
        if random.randint(0,50)==0:
            for peer_id in self.peers:
                if str(peer_id)+'request_block'+str(block_hash) in self.request_list:
                    if time.time()-self.request_list[str(peer_id)+'request_block'+str(block_hash)]>5:
                        self.send_message(peer_id, 'request_block', {'block_hash': block_hash})
                        self.request_list[str(peer_id)+'request_block'+str(block_hash)]=time.time()
                else:
                    self.request_list[str(peer_id)+'request_block'+str(block_hash)]=time.time()
                    self.send_message(peer_id, 'request_block', {'block_hash': block_hash})
        else:
            peer_id= voter
            if str(peer_id)+'request_block'+str(block_hash) in self.request_list:
                if time.time()-self.request_list[str(peer_id)+'request_block'+str(block_hash)]>5:
                    self.send_message(peer_id, 'request_block', {'block_hash': block_hash})
                    self.request_list[str(peer_id)+'request_block'+str(block_hash)]=time.time()
            else:
                self.request_list[str(peer_id)+'request_block'+str(block_hash)]=time.time()
                self.send_message(peer_id, 'request_block', {'block_hash': block_hash})

    def request_vote(self, vote_hash,voter):
        """Request a vote from peers."""
        if random.randint(0,50)==0:
            for peer_id in self.peers:
                if str(peer_id)+'request_vote'+str(vote_hash) in self.request_list:
                    if time.time()-self.request_list[str(peer_id)+'request_vote'+str(vote_hash)]>5:
                        self.send_message(peer_id, 'request_vote', {'vote_hash': vote_hash})
                        self.request_list[str(peer_id)+'request_vote'+str(vote_hash)]=time.time()
                else:
                    self.request_list[str(peer_id)+'request_vote'+str(vote_hash)]=time.time()
                    self.send_message(peer_id, 'request_vote', {'vote_hash': vote_hash})
        else:
            peer_id= voter
            if (str(peer_id)+'request_vote'+str(vote_hash)) in self.request_list:
                if time.time()-self.request_list[str(peer_id)+'request_vote'+str(vote_hash)]>5:
                    self.send_message(peer_id, 'request_vote', {'vote_hash': vote_hash})
                    self.request_list[str(peer_id)+'request_vote'+str(vote_hash)]=time.time()
            else:
                self.request_list[str(peer_id)+'request_vote'+str(vote_hash)]=time.time()
                self.send_message(peer_id, 'request_vote', {'vote_hash': vote_hash})

    def handle_block_request(self, requester_id, block_hash):
        """Send the requested block to the requester."""
        block = self.block_hash_map.get(block_hash)
        if block:
            self.send_message(requester_id, 'block', serialize_block(block))

    def handle_vote_request(self, requester_id, vote_hash):
        """Send the requested vote to the requester."""
        vote = self.vote_hash_map.get(vote_hash)
        if vote:
            self.send_message(requester_id, 'vote', serialize_vote(vote))

    def store_vote(self, vote):
        self.vote_hash_map[vote.get_vote_hash()] = vote

    def create_block(self, previous_block):
        return Block(
            block_hash=f"block_{self.node_id}_{previous_block.height + 1}_{random.randint(0, 9999)}",
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
            if latest_block_candidate.height+1 in self.blockchain:
                for  other_block in self.blockchain[latest_block_candidate.height+1]:
                    if other_block.previous_block_hash ==latest_block_candidate.block_hash:
                        return None # someone already proposed a block of the same height at this branch.
            new_block = self.create_block(latest_block_candidate)
            self.blockchain.setdefault(new_block.height, []).append(new_block)
            self.block_hash_map[new_block.block_hash] = new_block
            print(f"[{time.time()}]",f"Node {self.node_id} proposed block {new_block.block_hash} at consensus height {new_block.height} extending from {latest_block_candidate.block_hash}")
            # Broadcast the block to all peers
            block_json = serialize_block(new_block)
            for peer_id in self.peers:
                self.send_message(peer_id, 'block', block_json)
            return new_block  # Return block to broadcast to other nodes
        return None

    def adversary_propose_block(self):
        """Adversary randomly proposes a block, without considering leader rules."""
        # Randomly select a block from the entire blockchain to extend
        with self.lock:
            all_blocks = [block for height_blocks in self.blockchain.values() for block in height_blocks]
            if not all_blocks:
                # If there are no blocks (other than the genesis block), extend from the genesis block
                latest_block_candidate = self.genesis_block
            else:
                # Choose a random block from the blockchain to extend from
                latest_block_candidate = random.choice(all_blocks)
            # Create a new block extending from the randomly chosen block
            new_block = self.create_block(latest_block_candidate)
            # Add the new block to the blockchain
            self.blockchain.setdefault(new_block.height, []).append(new_block)
            self.block_hash_map[new_block.block_hash] = new_block
            print(f"[{time.time()}]",f"Adversary node {self.node_id} proposed block {new_block.block_hash} at consensus height {new_block.height}")
            # Adversary sends to a subset of peers
            block_json = serialize_block(new_block)
            num_peers = len(self.peers)
            if num_peers > 0:
                sample_size = random.randint(0, num_peers)
                adversary_subset = random.sample(list(self.peers.keys()), sample_size)
            else:
                adversary_subset = []
            for peer_id in adversary_subset:
                self.send_message(peer_id, 'block', block_json)
            return new_block  # Return block to broadcast to other nodes
    def voting (self):
        while not self.stop_event.is_set():
            self.cast_vote()
            time.sleep(0.5)
    
    def proposing_block (self):
        last_done=0
        while not self.stop_event.is_set():
            te=time.time() // self.delta_bi
            if te %self.num_nodes==self.node_id and last_done!=te:
                self.propose_block()
                last_done=te

    def cast_vote(self):
        """
        Cast a vote at the given vote height based on N-f compliant votes from previous heights.
        """
        vote_height = self.vote_height + 1

        if vote_height == 0:
            # At height 0, vote for the Genesis block
            genesis_vote = Vote(0, self.genesis_block.block_hash, self.node_id, [])
            genesis_vote.block = self.genesis_block
            genesis_vote.block_hash_map = self.block_hash_map
            self.vote_height = vote_height
            print(f"[{time.time()}]",f"Node {self.node_id} cast a vote for the Genesis block at vote height 0.")
            # Broadcast the vote
            vote_json = serialize_vote(genesis_vote)
            for peer_id in self.peers:
                self.send_message(peer_id, 'vote', vote_json)
            genesis_vote.build_vote_graph()
            genesis_vote.calculate_vote_count()
            self.store_vote(genesis_vote)
            self.received_votes.append(genesis_vote)
            return genesis_vote  # Return vote to broadcast to other nodes

        block_to_vote_for = self.find_block_to_vote_for(vote_height)
        if block_to_vote_for is not None:
            previous_vote_height = vote_height - 1
            compliant_votes = [vote for vote in self.received_votes if vote.height == previous_vote_height and vote.is_compliant_vote()]

            if len(compliant_votes) >= self.N_f_votes_required:
                # Create the list of previous vote hashes
                previous_vote_hashes = [vote.get_vote_hash() for vote in compliant_votes]
                # Cast the vote linking to the N-f compliant votes
                new_vote = Vote(vote_height, block_to_vote_for.block_hash, self.node_id, previous_vote_hashes)
                new_vote.block = block_to_vote_for
                new_vote.previous_votes = compliant_votes
                new_vote.block_hash_map = self.block_hash_map
                self.store_vote(new_vote)
                self.received_votes.append(new_vote)
                self.vote_height = vote_height
                 # Broadcast the vote
                vote_json = serialize_vote(new_vote)
                for peer_id in self.peers:
                    self.send_message(peer_id, 'vote', vote_json)
                new_vote.build_vote_graph()
                new_vote.calculate_vote_count()
                print(f"[{time.time()}]",f"Node {self.node_id} cast a vote for block {block_to_vote_for.block_hash} at vote height {vote_height}. This vote is linked to the following nodes' vote at vote height {vote_height-1}:",[item.voter for item in new_vote.previous_votes], f"The VC of this vote is:{new_vote.VC}")
                return new_vote  # Return vote to broadcast to other nodes
            else:
                print(f"[{time.time()}]",f"Node {self.node_id} is waiting for more compliant votes at vote height {previous_vote_height}.")
        else:
            pass
            #print(f"[{time.time()}]",f"Node {self.node_id} is waiting for enough information to vote at height {vote_height}.")
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

    def get_second_most_supported_blocks(self, starting_height, consensus_height):
        """
        Determine one block at the given consensus height that is part of the second most supported branch.
        If there are multiple branches with the same vote count, randomly select one, provided that these branches have blocks at consensus_height.
        """
        # Step 1: Filter compliant votes at the given consensus height
        compliant_votes = [vote for vote in self.received_votes if vote.block.height == consensus_height and vote.is_compliant_vote()]
        if len(compliant_votes) < self.N_f_votes_required:
            return []

        # Step 2: Create the virtual vote using compliant votes
        virtual_vote = self.create_virtual_vote(compliant_votes)

        # Step 3: Track both most and second most supported blocks at each level
        most_supported_blocks_at_each_level = {}
        second_most_supported_blocks_at_each_level = {}
        highest_vote_count_at_height = {}
        second_highest_vote_count_at_height = {}

        # Step 4: Find the blocks with the most and second most support at starting_height
        for block_hash, vote_count in virtual_vote.VC.items():
            block = self.block_hash_map.get(block_hash)
            if block.height == starting_height:
                if block.height not in highest_vote_count_at_height or vote_count > highest_vote_count_at_height[block.height]:
                    second_highest_vote_count_at_height[block.height] = highest_vote_count_at_height.get(block.height, 0)
                    second_most_supported_blocks_at_each_level[block.height] = most_supported_blocks_at_each_level.get(block.height, [])

                    highest_vote_count_at_height[block.height] = vote_count
                    most_supported_blocks_at_each_level[block.height] = [block]
                elif vote_count == highest_vote_count_at_height[block.height]:
                    most_supported_blocks_at_each_level[block.height].append(block)
                elif vote_count > second_highest_vote_count_at_height[block.height]:
                    second_highest_vote_count_at_height[block.height] = vote_count
                    second_most_supported_blocks_at_each_level[block.height] = [block]
                elif vote_count == second_highest_vote_count_at_height[block.height]:
                    second_most_supported_blocks_at_each_level[block.height].append(block)

        if starting_height == consensus_height:
            second_most_supported_blocks = second_most_supported_blocks_at_each_level.get(starting_height, [])
            return [random.choice(second_most_supported_blocks)] if second_most_supported_blocks else []

        # Step 5: Trace down the second most supported branches by checking ancestors at each height
        for h in range(starting_height + 1, consensus_height + 1):
            most_supported_blocks_at_each_level[h] = []
            second_most_supported_blocks_at_each_level[h] = []
            highest_vote_count_at_height[h] = 0
            second_highest_vote_count_at_height[h] = 0

            # For each block in the previous level, find its valid children
            for parent_block in second_most_supported_blocks_at_each_level[h - 1]:
                for block_hash, vote_count in virtual_vote.VC.items():
                    block = self.block_hash_map.get(block_hash)
                    if block and block.height == h and block.previous_block_hash == parent_block.block_hash:
                        if vote_count > highest_vote_count_at_height[h]:
                            second_highest_vote_count_at_height[h] = highest_vote_count_at_height[h]
                            second_most_supported_blocks_at_each_level[h] = most_supported_blocks_at_each_level.get(h, [])

                            highest_vote_count_at_height[h] = vote_count
                            most_supported_blocks_at_each_level[h] = [block]
                        elif vote_count == highest_vote_count_at_height[h]:
                            most_supported_blocks_at_each_level[h].append(block)
                        elif vote_count > second_highest_vote_count_at_height[h]:
                            second_highest_vote_count_at_height[h] = vote_count
                            second_most_supported_blocks_at_each_level[h] = [block]
                        elif vote_count == second_highest_vote_count_at_height[h]:
                            second_most_supported_blocks_at_each_level[h].append(block)

            # If no valid blocks are found at this level, the second most supported branch is incomplete
            if not second_most_supported_blocks_at_each_level[h]:
                return []

        # Step 6: Select one block from the second most supported branch at consensus_height randomly
        second_most_supported_blocks = second_most_supported_blocks_at_each_level.get(consensus_height, [])
        return [random.choice(second_most_supported_blocks)] if second_most_supported_blocks else []

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
        virtual_vote = Vote(height=compliant_votes[0].height + 1, block_hash=None, voter=self.node_id, previous_vote_hashes=[vote.get_vote_hash() for vote in compliant_votes])
        virtual_vote.previous_votes = compliant_votes
        virtual_vote.block_hash_map = self.block_hash_map  # Provide access to block hash map
        virtual_vote.build_vote_graph()
        virtual_vote.calculate_vote_count()
        return virtual_vote

    def is_consensus_reached(self, k_prime, k):
        """
        Determine if consensus is reached on a block at height k_prime based on votes at height k.
        """
        while k>k_prime+1:
            compliant_votes = [vote for vote in self.received_votes if vote.block.height == k and vote.is_compliant_vote()]
            if len(compliant_votes) < self.N_f_votes_required:
                continue
            for item in compliant_votes:
                most_supported_block, second_most_supported_block,v1,v2 = self.get_top_two_supported_branches(k_prime, k-1, item.G)
                if not most_supported_block:
                    continue
                
                missing_votes_at_k_prime = self.missing_votes_count(k_prime,item.G)
                missing_votes_at_k_1 = self.missing_votes_count(k-1,item.G)
                potential_double_votes = (k  - k_prime) * self.f
               # print (v1, v1 - potential_double_votes,  v2 + missing_votes_at_k_1 - missing_votes_at_k_prime - 1)
                if v1 - potential_double_votes > v2 + missing_votes_at_k_1 - missing_votes_at_k_prime - 1:
                    print(f"[{time.time()}]",f"Node {self.node_id} thinks Consensus reached on block {most_supported_block} at height {k} because {most_supported_block} got {v1} supports, {second_most_supported_block} got {v2} supports, the missing votes are {missing_votes_at_k_1 - missing_votes_at_k_prime}.")
                    self.most_recent_accepted = most_supported_block
                    return most_supported_block
            k-=1
        return False

    def get_top_two_supported_branches(self, starting_height,consensus_height, G):
        """
        Find the most supported and second most supported branches starting from a given starting height.
        The most supported branch has the highest vote count at the starting height, and at each subsequent height,
        the ancestors in the chain also need to have the highest vote count among their siblings. The second-most branch
        has the second-highest vote count.
        """
        # Step 1: Filter compliant votes at the given starting height
        compliant_votes = [vote for vote in G if vote.block.height == consensus_height and vote.is_compliant_vote()]
        if len(compliant_votes) < self.N_f_votes_required:
            return None, None, None,None

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
        return block_support_at_starting_height[0][0], block_support_at_starting_height[1][0],block_support_at_starting_height[0][1], block_support_at_starting_height[1][1]

    def get_block_by_hash(self, block_hash):
        return self.block_hash_map.get(block_hash, None)

    def missing_votes_count(self, height, G):
        total_votes = self.num_nodes * (height+1)
        received_votes = sum(1 for vote in G if vote.height <= height)
        return total_votes - received_votes

    def stop(self):
        self.stop_event.set()
        if self.server_socket:
            self.server_socket.close()

# Simulation Code

def simulate_voting_process(num_nodes=7, num_rounds=10, block_interval=10):
    genesis_block = Block(block_hash="genesis", height=0, previous_block_hash=None, proposer_id=None)
    f = 2

    # Create nodes
    nodes = [Node(node_id=i, num_nodes=num_nodes, f=f, genesis_block=genesis_block, delta_bi=block_interval, base_port=6000) for i in range(num_nodes)]

    # Set up peers
    for node in nodes:
        for other_node in nodes:
            if other_node.node_id != node.node_id:
                node.peers[other_node.node_id] = (other_node.host, other_node.port)

    # Start nodes' servers (already started in __init__)
    stop_event = threading.Event()
    print ("Setup completed")
    for node in nodes:
        node.start_working()

    def propose_blocks():
        """Thread function to propose blocks at fixed time intervals."""
        block_round = 0
        while not stop_event.is_set():
            #print(f"[{time.time()}]",f"\n--- Block Interval {block_round + 1} ---")

            # Each node proposes blocks
            for node in nodes:
             #   node.propose_block()
                if random.randint(1,100)==1:
                    node.adversary_propose_block()

           # block_round += 1
            time.sleep(block_interval)

    def check_consensus():
        """Thread function to check for consensus."""
        k_prime = 0
        first_time = {}
        while not stop_event.is_set():
            for node in nodes:
                if node.latest_vote_height<=k_prime:
                    continue
                consensus = node.is_consensus_reached(k_prime, node.latest_vote_height)
                if consensus and consensus not in first_time:
                    #print(f"[{time.time()}]",f"Consensus for height {k_prime} has been reached: {consensus} at vote height {node.latest_vote_height}")
                    k_prime += 1
                    first_time[consensus] = True
                    break
            if k_prime >num_rounds:
                os._exit()
            time.sleep(5)

    # Start threads
  #  block_thread = threading.Thread(target=propose_blocks)
    #vote_thread = threading.Thread(target=cast_votes)
    consensus_thread = threading.Thread(target=check_consensus)
    #block_thread.start()
    #vote_thread.start()
    consensus_thread.start()

 #   try:
 #       time.sleep(num_rounds * block_interval)
 #   finally:
 #       # Stop the threads
 #       stop_event.set()
  #      for node in nodes:
  #          node.stop()
  #      block_thread.join()
  #      vote_thread.join()
  #      consensus_thread.join()

# Run the simulation
if __name__ == "__main__":
    simulate_voting_process()
