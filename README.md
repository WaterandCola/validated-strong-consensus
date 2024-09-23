**Validated Strong Consensus Protocol Small Toy Experiment and Visualization**

This repository contains a simplified Python experiment and web-based visualization tool for the Validated Strong Consensus protocol. The logic for voting and consensus-reaching is exactly the same as the code we used for the experiments; however, other features have been simplified for illustration purposes. In particular, the designs and optimizations in Sec. 3.6.1 and Sec. 3.7 are not implemented in this toy experiment. Additionally, we do not use a leader-collecting system to optimize complexity.

The block interval is set to 10 seconds, meaning the honest nodes will check if they should propose a block (as a leader node) in every 10 seconds. 
The nodes will vote as fast as they can (when collected N-f votes from the last vote height).


 **Contents**
- `vote.py`: The simulation script for the consensus protocol (using threads to simulate nodes).
- `vote_multiprocesses.py`: The simulation script for the consensus protocol (using processes to simulate nodes).
- `index.html`: The web-based visualization of the consensus process.
- `image.png`: A screenshot of the visualized output for an example consensus scenario.
- `Code Listing 1.ipynb`: Code Listing 1 and how Figure 4 of the paper was generated.
- `README.md`: This document.

 **How to Use**

 **Run the Simulation**
To simulate the voting process of our protocol, you can run the `vote.py` script. This will set up a toy example of a network of `N=7` nodes, with a fault tolerance `f=2`, running our protocol. No time-bounds were set for the voting process in asynchronous environments and you can visualize the consensus process. In this process, all 7 nodes behave normally. This is because if adversary cast non-compliant votes, their votes are not processed, resulting in 5 active honest nodes and two silent adversaries. Since each node only votes after receiving $N-f$ votes from the previous height, having only 5 active honest nodes leads to very fast branch alignment (since each one will have the same view after one voting round), making it less interesting for illustration.

**Steps:**
1. Run the Python simulation.
   ```bash
   python3 vote.py > output.txt
   ```
   or
   ```bash
   python3 vote_multiprocesses.py > output.txt
   ```
As the number of threads allowed in a process is limited, simulating nodes using threads (as node uses multiple threads) can be considered as simulating nodes in a slow network, where some nodes experience delays. In contrast, simulating nodes using processes can provide more parallelism, which can be seen as simulating nodes in a network with good conditions. However, in either case, there is no time-bound for reaching consensus.
    
2. The simulation will output logs describing the blocks proposed and the votes cast at each round.

 **Visualize the Consensus Process**
After running the simulation, you can visualize the consensus process using the web-based tool included in this repository.

1. Open `index.html` in a web browser.
2. You can either manually paste the log from the Python script into the "Logs Input" area or generate new logs by running the simulation.
3. The graph will display the blocks and votes, showing the paths taken by each node to reach consensus.
   - Blocks with the most votes at each level are highlighted in green, representing consensus.
   - Hover over votes to see details about the vote's ID and the node that cast it.
4. Adjust the input to replay different scenarios by modifying the log data.

 **Example**
To explore the simulation and its visual representation, run the example provided. The default scenario will demonstrate how nodes in the network propose blocks, cast votes, and eventually reach consensus through Validated Strong BFT Consensus Protocol.

 **Explanation of the Visualization (Figure)**
The provided screenshot (`image.png`) demonstrates the output of the consensus visualization tool.

1. **Blocks** (represented by squares) are proposed at each height by the nodes.
2. **Votes** (represented by circles) are cast by nodes, with each vote supporting a block.
3. The **green block** shows the block that has reached consensus, having received the majority of votes from the nodes.
4. Hovering over votes and blocks in the tool will display details of the vote graph associated with the vote as well as the vote count (VC) derived.
5. The connections between the votes and blocks visualize how votes accumulate, ultimately leading to consensus on a specific block.


We hope you enjoy our paper and have a lovely reviewing expereince!

The auhtors
