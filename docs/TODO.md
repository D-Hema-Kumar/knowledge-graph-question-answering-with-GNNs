## Programming changes
    - [ ] Refactor everything
    - [ ] Put everything into a class (including context; check the [KGCIF](https://github.tools.sap/business-process-intelligence/psmg-imkg-kgcif) repo for really sophistication via pydantic)
    - [ ] Make sure experiments are called via main; decouple experiment definition from execution as much as possible
    - [ ] Write some tests with pytest to make sure your experiments do not get broken via code changes
    - [ ] Store results for experiments in disk

## Experiments to Try out with current Neural Nets
- [ ] Simple BinaryClassification with MLP
- [ ] Simple BinaryClassification with GCN
- [ ] Classify instances and retrieve embedding to give to Jan
- [ ] QA simple BinaryClassification and MLP
- [ ] QA simple BinaryClassification and MLP with masks
- [ ] QA simple BinaryClassification and GCN
- [ ] QA simple BinaryClassification and GCN with masks 

## Neural Nets functionality to add
- [ ] Add edges to Graphs
- [ ] Use RGCN
- [ ] Make masks depend on subgraph generate and not on random Node selection

## Others
- [ ] Change the Goldstandard to include non "Type" questions