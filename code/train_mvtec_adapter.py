"""Fine-tune only the TextPromptAdapter on top of an existing AnomalyGPT checkpoint.

Usage (from code/ directory):
    deepspeed --include localhost:0,1 --master_port 28400 train_mvtec_adapter.py \
        --model openllama_peft \
        --stage 1 \
        --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
        --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
        --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
        --anomalygpt_ckpt_path ./ckpt/train_mvtec/pytorch_model.pt \
        --max_tgt_len 512 \
        --data_path ../data/pandagpt4_visual_instruction_data.json \
        --image_root_path /workspace/Master-Thesis/data/images/ \
        --save_path ./ckpt/train_mvtec_adapter/ \
        --log_path ./ckpt/train_mvtec_adapter/log/ \
        --epochs 3
"""

from header import *
from datasets import *
from model import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='adapter fine-tuning parameters')
    parser.add_argument('--model', type=str)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--imagebind_ckpt_path', type=str)
    parser.add_argument('--vicuna_ckpt_path', type=str)
    parser.add_argument('--delta_ckpt_path', type=str)
    parser.add_argument('--anomalygpt_ckpt_path', type=str)
    parser.add_argument('--max_tgt_len', type=int)
    parser.add_argument('--stage', type=int)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--image_root_path', type=str)
    parser.add_argument('--epochs', type=int, default=3)
    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args):
    args['root_dir'] = '../'
    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])


def build_directory(path):
    os.makedirs(path, exist_ok=True)


class AdapterAgent:
    """DeepSpeed agent that only trains the TextPromptAdapter parameters."""

    def __init__(self, model, args):
        self.args = args
        self.model = model

        delta_ckpt = torch.load(args['delta_ckpt_path'], map_location='cpu')
        self.model.load_state_dict(delta_ckpt, strict=False)

        if args.get('anomalygpt_ckpt_path'):
            anomaly_ckpt = torch.load(args['anomalygpt_ckpt_path'], map_location='cpu')
            self.model.load_state_dict(anomaly_ckpt, strict=False)
            print(f'[!] Loaded AnomalyGPT checkpoint from {args["anomalygpt_ckpt_path"]}')

        for name, param in self.model.named_parameters():
            param.requires_grad = False

        for name, param in self.model.text_adapter.named_parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f'[Adapter-only] trainable params: {trainable} || all params: {total} || trainable%: {trainable/total*100:.6f}')

        ds_params = json.load(open(self.args['ds_config_path']))
        ds_params['scheduler']['params']['total_num_steps'] = self.args['total_steps']
        ds_params['scheduler']['params']['warmup_num_steps'] = max(10, int(self.args['total_steps'] * self.args['warmup_rate']))
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args),
        )

    def train_model(self, batch, current_step=0, pbar=None):
        self.ds_engine.module.train()
        loss, mle_acc = self.ds_engine(batch)
        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        pbar.update(1)
        if self.args['local_rank'] == 0 and self.args.get('log_path') and current_step % self.args.get('logging_step', 10) == 0:
            elapsed = pbar.format_dict['elapsed']
            rate = pbar.format_dict['rate']
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            logging.info(f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}')
        return mle_acc * 100

    def save_model(self, path, current_step):
        if int(os.environ.get('RANK', '0')) != 0:
            return
        from collections import OrderedDict
        checkpoint = OrderedDict()
        for k, v in self.ds_engine.module.named_parameters():
            if v.requires_grad:
                checkpoint[k] = v.data.cpu().clone()
        os.makedirs(path, exist_ok=True)
        torch.save(checkpoint, f'{path}/adapter_model.pt')
        print(f'[!] Adapter checkpoint saved to {path}/adapter_model.pt')


def main(**args):
    config_env(args)
    args['ds_config_path'] = 'dsconfig/openllama_peft_stage_adapter.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w',
        )

    train_data, train_iter, sampler = load_mvtec_dataset(args)
    train_data_sft, train_iter_sft, sampler = load_sft_dataset(args)

    length = args['epochs'] * len(train_data) // args['world_size'] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = 2 * args['epochs'] * len(train_data) // dschf.config['train_batch_size']
    args['total_steps'] = total_steps

    model = OpenLLAMAPEFTModel(**args)
    agent = AdapterAgent(model, args)
    torch.distributed.barrier()

    pbar = tqdm(total=2 * length)
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        for batch, batch_sft in zip(train_iter, train_iter_sft):
            agent.train_model(batch, current_step=current_step, pbar=pbar)
            del batch
            agent.train_model(batch_sft, current_step=current_step, pbar=pbar)
            del batch_sft
            current_step += 1
            if current_step % 500 == 0:
                torch.distributed.barrier()
                agent.save_model(args['save_path'], current_step)
        torch.distributed.barrier()
        agent.save_model(args['save_path'], current_step)


if __name__ == '__main__':
    args = parser_args()
    args = vars(args)
    args['layers'] = [7, 15, 23, 31]
    main(**args)
