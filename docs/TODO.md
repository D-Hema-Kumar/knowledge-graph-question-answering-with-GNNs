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
- [ ] QA simple BinaryClassification and MLP
- [ ] QA simple BinaryClassification and MLP with masks
- [ ] QA simple BinaryClassification and GCN -- Done
- [ ] QA simple BinaryClassification and GCN with masks -- Done

## Neural Nets functionality to add
- [ ] Add edges to Graphs -- Done
- [ ] Use RGCN -- Done
- [ ] Make masks depend on subgraph generate and not on random Node selection -- Done

## Others
- [ ] Change the Goldstandard to include non "Type" questions -- Done
- Add more question
- Get different raondom samples from 