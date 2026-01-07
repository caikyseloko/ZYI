```markdown
# zyi — Text-to-Image do zero (arquitetura nova)  

Resumo
- Nome do projeto/modelo: zyi (ZYI-Net)
- Arquitetura: TextEncoder (Transformer) -> Composer (layout multiescala) -> FlowPatch (normalizing flow por patch) -> ImplicitPainter (MLP implícito por patch).
- Sem GAN, sem UNet, sem diffusion models.
- Dataset recomendado: COCO 2014 (captions originais). Opcionalmente usar BLIP (pré-treinado) para gerar/ajustar captions.

Objetivo
- Fornecer código Python (scripts, módulos) para treinar do zero em Colab (arquivo .py, não notebook). O pipeline é pensado para experimentos em 128×128.

Como usar no Colab (passo a passo resumido)
1) Suba estes arquivos ao seu repositório GitHub (ex.: caikyseloko/zyi). No Colab rode:
   !git clone https://github.com/caikyseloko/zyi.git
   cd zyi

2) Instale dependências:
   pip install -r requirements.txt

3) Baixe COCO 2014:
   - Baixe train2014 e val2014 images + annotations (captions) do site COCO.
   - Coloque as pastas localmente (ou monte Google Drive).

4) Prepare dataset 128×128:
   python scripts/prepare_coco.py --coco-ann /path/to/annotations/captions_train2014.json --images-dir /path/to/train2014 --out-dir data/coco128 --use-blip False
   (Use --use-blip True se quiser regenerar captions com BLIP)

5) Treino (exemplo rápido no Colab, small subset):
   python train_colab.py --data-dir data/coco128 --epochs 20 --batch-size 16 --checkpoint-dir /content/drive/MyDrive/zyi_checkpoints

6) Inferência:
   python eval.py --checkpoint /path/to/checkpoint.pth --vocab data/coco128/vocab.json --text "um gato dormindo" --out out.png

Observações
- Código é um protótipo funcional. Em produção ajuste hiperparâmetros, vectorização, mixed precision, sharding.
- FlowPatch aqui é uma implementação simples de RealNVP por patch; você pode aprimorar suas camadas, permutações e condicionamentos.
- Se quiser que eu faça a adição automática ao seu repositório (PR ou commit), autorize com os comandos git (ou me diga para gerar os comandos).

Licença: use conforme sua política. Não inclua imagens COCO no repositório por licença — baixe e prepare localmente.
```
