# Hybrid-RAG-POC-with-Earning-Calls
This project was inspired by the paper HybridRAG: Integrating Knowledge Graphs and Vector Retrieval
Augmented Generation for Efficient Information Extraction for HSBC Ai Engineer Interview.

## Decisive Differences between VectorDB RAG alone and My version of Hybird RAG (Without separated Cypher query)
When asking questions about analysis of earnings calls when the right chunk of data wasn't selected, Vector DB seems having much more difficulties on the task while My version of Hybrid RAG is doing a much better job.

**Question:**
Can you compare the plan for both HSBC and DBS have in 2024?  Give me a short and simple answer.
**Vector DB RAG**|**My version of Hybrid RAG**
-|-
Approximate tokens size: 1469|Approximate tokens size: 1682
Based on the provided document, here's a simple comparison between HSBC and DBS' 2024 plans: **HSBC:** Reconfirms guidance: + Mid-teens return on tangible equity excluding notable items + Banking NII of at least $41 billion + ECLs of around 40 basis points + Cost growth limited to circa 5% on a target basis + 50% dividend payout ratio Focused on: + Continuing momentum and execution of strategy + Development of wealth, transaction banking capabilities, and global wholesale banking **DBS:** Unfortunately, there is no mention of DBS' plans for 2024 in the provided document. The conversation only discusses HSBC's performance and guidance for the year. If you'd like to provide more information about DBS' 2024 plans, I can help compare them with HSBC's goals.|Here is a short and simple comparison of HSBC and DBS plans for 2024: **Similarities:** Both banks prioritize shareholder value creation through share buybacks and dividend distributions. They focus on strategic growth areas like wealth management, global wholesale banking, and transaction banking. Both emphasize cost management and efficiency to maintain profitability. **Differences:** **Income Growth:** HSBC expects 2024 NII guidance of at least $41 billion, while DBS anticipates a 6-7% income growth for the near term. **Capital Management:** DBS aims to increase its capital ratio by 70 basis points, whereas HSBC's focus is on enhanced hedging strategies and asset quality management. **Loan Growth:** DBS experiences strong loan growth driven by high demand in the commodity sector, whereas HSBC focuses on expanding its Wealth division and Global Payments Solutions. **Risk Management:** DBS maintains excess GP reserves to cushion against potential stress, while HSBC reaffirmed its 2024 expected credit loss (ECL) guidance at around 40 basis points.

I think this bring out other issues of Vector DB RAG, which is when the query require doing multiple documents analysis, the selected chunks may not be evenly distributed and causing the LLM not able to retrieve required information effective.

For My version of Hybrid RAG, it will be more even since even if the Vector Search is not even, but after pre and post information added to the chunks, the LLM will be able to grasp the whole idea better and do a better job.

## Hybrid RAG
I created some nodes call parameters and they will be connected to the banks with the specific documents that is related such as Profit, Cost, Challenge, Opportunity, Plan, Significant one-time gain or loss, dividend policy.
I also created another function called parameters extract for extracting the key value required for Cypher Query. This way we can use the Knowledge Graph more effectively without hardcoding the Cypher Queries to the system.

## Setup
If you need to try it, you will need:

Dify
Neo4j
Ollama
pytorch with CUDA

And also install the required libraries.