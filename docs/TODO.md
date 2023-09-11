## Programming changes
    - [ ] Refactor everything
    - [ ] Put everything into a class (including context; check the [KGCIF](https://github.tools.sap/business-process-intelligence/psmg-imkg-kgcif) repo for really sophistication via pydantic)
    - [ ] Make sure experiments are called via main; decouple experiment definition from execution as much as possible -- Done
    - [ ] Write some tests with pytest to make sure your experiments do not get broken via code changes
    - [ ] Store results for experiments in disk -- Done

## Experiments to Try out with current Neural Nets
- [ ] Simple BinaryClassification with MLP -- Done
- [ ] Simple BinaryClassification with GCN -- Done
- [ ] Classify instances and retrieve embedding to give to Jan -- Done
- [ ] QA using MLP
- [ ] QA using MLP with masks
- [ ] QA using GCN -- Done
- [ ] QA using GCN with masks -- Done
- [ ] QA using R-GCN with masks -- Done
- [ ] QA using R-GAT with masks
- [ ] QA using R-GAT with edge features


## Neural Nets functionality to add
- [ ] Add edges to Graphs -- Done
- [ ] Use RGCN -- Done
- [ ] Make masks depend on subgraph generate and not on random Node selection -- Done
- [ ] Implement GAT mechanism with edge features

## Others
- [ ] Change the Goldstandard to include non "Type" questions -- Done
- Add more questions.

## Optimizations

- [ ] refactor code: To store subgraphs in a file, read and train, might save time while training.
- [ ] Instead of predicting on all the nodes, predict on only the subgraph nodes.