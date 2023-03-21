# -*- coding: utf-8 -*-

import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, CTCLabelConverterForBaiduWarpctc, AttnLabelConverter, Averager
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

os.environ["CUDA_VISIBLE_DEVICES"] = "204"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(opt):
    """ dataset preparation """
    if not opt.data_filtering_off:
        print('Filtering the test containing characters which are not in opt.character')
        print('Filtering the test whose label is longer than opt.batch_max_length')
        # see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L130

    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        shuffle=True,  # 'True' to check test progress with test function.
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    log.write(valid_dataset_log)
    print('-' * 80)
    log.write('-' * 80 + '\n')
    log.close()
    
    """ model configuration """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(opt.character)
        else:
            converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()
    if opt.saved_model != '':
        print(f'loading pretrained model from {opt.saved_model}')

        if opt.FT:
            checkpoint = torch.load(opt.saved_model, map_location=device)
            # checkpoint = torch.load(opt.saved_model)
            checkpoint = {k: v for k, v in checkpoint.items() if
                          (k in model.state_dict().keys()) and (model.state_dict()[k].shape == checkpoint[k].shape)}
            for name in model.state_dict().keys():
                if name in checkpoint.keys():
                    model.state_dict()[name].copy_(checkpoint[name])
        else:
            model.load_state_dict(torch.load(opt.saved_model))

        '''
        if opt.FT:
            model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'), strict=False)
        else:
            model.load_state_dict(torch.load(opt.saved_model, map_location='cpu'))
        '''

    """ setup loss """
    if 'CTC' in opt.Prediction:
        if opt.baiduCTC:
            # need to install warpctc. see our guideline.
            from torch import nn
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
            # from warpctc_pytorch import CTCLoss
            # criterion = CTCLoss()
        else:
            criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.exp_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start test """
    start_iter = 0
    if opt.saved_model != '':
        try:
            start_iter = int(opt.saved_model.split('_')[-1].split('.')[0])
            print(f'continue to train, start_iter: {start_iter}')
        except:
            pass

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    iteration = start_iter

    while(True):
        # train part
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format
                cost = criterion(preds, text, preds_size, length) / batch_size
            else:
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

        else:
            preds = model(image, text[:, :-1])  # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        # test part
        if (iteration + 1) % opt.valInterval == 0 or iteration == 0: # To see test progress, we also conduct test when 'iteration == 0'
            elapsed_time = time.time() - start_time
            # for log
            with open(f'./saved_models/{opt.exp_name}/log_train.txt', 'a') as log:
                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, confidence_score, labels, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                # test loss and test loss
                loss_log = f'[{iteration+1}/{opt.num_iter}] Train loss: {loss_avg.val():0.5f}, Valid loss: {valid_loss:0.5f}, Elapsed_time: {elapsed_time:0.5f}'
                loss_avg.reset()

                current_model_log = f'{"Current_accuracy":17s}: {current_accuracy:0.3f}, {"Current_norm_ED":17s}: {current_norm_ED:0.2f}'

                # keep best accuracy model (on valid dataset)
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_accuracy.pth')
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_norm_ED.pth')
                best_model_log = f'{"Best_accuracy":17s}: {best_accuracy:0.3f}, {"Best_norm_ED":17s}: {best_norm_ED:0.2f}'

                loss_model_log = f'{loss_log}\n{current_model_log}\n{best_model_log}'
                print(loss_model_log)
                log.write(loss_model_log + '\n')

                # show some predicted results
                dashed_line = '-' * 80
                head = f'{"Ground Truth":25s} | {"Prediction":25s} | Confidence Score & T/F'
                predicted_result_log = f'{dashed_line}\n{head}\n{dashed_line}\n'
                for gt, pred, confidence in zip(labels[:5], preds[:5], confidence_score[:5]):
                    if 'Attn' in opt.Prediction:
                        gt = gt[:gt.find('[s]')]
                        pred = pred[:pred.find('[s]')]

                    predicted_result_log += f'{gt:25s} | {pred:25s} | {confidence:0.4f}\t{str(pred == gt)}\n'
                predicted_result_log += f'{dashed_line}'
                print(predicted_result_log)
                log.write(predicted_result_log + '\n')

        # save model per 1e+5 iter.
        if (iteration + 1) % 1e+5 == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.exp_name}/iter_{iteration+1}.pth')

        if (iteration + 1) == opt.num_iter:
            print('end the test')
            sys.exit()
        iteration += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to test dataset')
    parser.add_argument('--valid_data', required=True, help='path to test dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=10000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=50, help='Interval between each test')
    parser.add_argument('--saved_model', default='', help="path to model to continue test")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--baiduCTC', action='store_true', default=True, help='for data_filtering_off mode')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='/',
                        help='select test data (default is MJ-ST, which means MJ and ST used as test data)')
    parser.add_argument('--batch_ratio', type=str, default='1',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=10, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZㆍ가각간갇갈갉감갑값갓갔강갖같갚개객갤갭갯갱갸걀거걱건걷걸검겁것겉게겐겔겟겠겨격겪견결겸겹겼경곁계고곡곤곧골곰곱곳공곶과곽관괄괌광괘괜괭괴괼굉교굘구국군굳굴굵굶굽굿궁궂궈권궐궤귀귄귈규균귤그극근글긁금급긋긍기긴길김깁깃깊까깍깎깐깔깜깝깡깥깨깬꺼꺾껄껌껍껏껐껑께껴꼈꼬꼭꼴꼼꼽꽁꽂꽃꽈꽉꽝꽤꾀꾸꾼꿀꿇꿈꿔꿨꿰뀌뀐뀔끄끈끊끌끓끔끗끝끼낀낄낌나낙낚난날낡남납낫났낭낮낯낱낳내낸낼냄냅냇냈냉냐냥너넉넌널넓넘넛넣네넥넨넬넵넷녀녁년념녔녕녘녜녠노녹논놀놈놉농높놓놔놨뇌뇨뇰뇽누눅눈눌눔눕눙눠뉘뉜뉴늄느늑는늘늙늠능늦늪늬니닉닌닐님닙닛닝다닥닦단닫달닭닮닳담답닷당닻닿대댁댄댈댐댑댓더덕던덜덟덤덥덧덩덫덮데덱덴델뎀뎅뎌뎬도독돈돋돌돔돕돗동돛돼됐되된될됨됩두둑둔둘둠둡둥둬뒀뒤뒷듀듈듐드득든듣들듬듭듯등디딕딘딛딜딥딧딩딪따딱딴딸땀땃땄땅때땐땡떠떡떤떨떻떼또똑똔똘똥뚜뚝뚫뚱뛰뛴뛸뜀뜨뜩뜬뜯뜰뜸뜻띄띈띠띤라락란랄람랍랏랐랑랗래랙랜랠램랩랫랬랭랴략랸량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄로록론롤롬롭롯롱롸뢰룀료룡루룩룬룰룸룹룻룽뤄뤘뤼뤽륀륄류륙륜률륨륭르륵른를름릅릇릉릎리릭린릴림립릿링마막만많맏말맑맘맙맛망맞맡맣매맥맨맬맴맵맷맹맺먀머먹먼멀멈멋멍메멕멘멜멤멧멩며멱면멸몄명몇모목몫몬몰몸몹못몽뫼묀묘무묵묶문묻물뭄뭇뭉뭐뭔뭘뮌뮐뮤뮬므믈믐미믹민믿밀밈밋밍및밑바박밖반받발밝밟밤밥밧방밭배백밴밸뱀뱃뱅뱉뱌버벅번벌범법벗벙벚베벡벤벨벰벳벵벼벽변별볍병볕보복볶본볼봄봅봇봉봐봤뵈뵙뵤부북분불붉붐붓붕붙뷔뷘뷰브븐블비빅빈빌빔빕빗빙빚빛빠빡빤빨빵빼빽뺀뺏뺑뺨뻐뻔뻗뻘뻤뼈뽀뽑뾰뿌뿐뿔뿜쁘쁜쁨삐사삭산살삶삼삽삿샀상새색샌샐샘샛생샤샨샬샴샵샷샹섀서석섞선섣설섬섭섯섰성세섹센셀셈셉셋셍셔션셜셤셨셰셴셸소속손솔솜솟송솽쇄쇠쇼숀숄숍숏숑수숙순술숨숫숭숱숲숴쉐쉬쉴쉼쉽슈슐슘슛슝스슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싼쌀쌈쌌쌍쌓써썩썬썰썸썹썼썽쎄쏘쏙쏜쏟쏠쏴쐈쐐쑈쑤쑥쑨쑹쓰쓴쓸씀씌씨씩씬씹씻씽아악안앉않알앓암압앗았앙앞애액앤앨앰앱앳앴앵야약얀얄얇얌얏양얕얘어억언얹얻얼얽엄업없엇었엉엌엎에엑엔엘엠엡엣엥여역엮연열엷염엽엿였영옅옆예옌옐옙옛오옥온올옭옮옳옴옵옷옹옻와왁완왈왓왔왕왜외왼요욕욘욜용우욱운울움웁웃웅워웍원월웜웠웨웬웰웸웹위윅윈윌윔윗윙유육윤율윳융으은을음읍응의이익인일읽잃임입잇있잉잊잎자작잔잖잘잠잡잣장잦재잭잼잿쟁쟈쟝저적전절젊점접젓정젖제젝젠젤젭젯젱져젼졌조족존졸좀좁종좇좋좌좡죄죠주죽준줄줌줍중줘줬쥐쥔쥘쥬즈즉즌즐즘즙증지직진질짊짐집짓징짖짙짚짜짝짠짤짧짱째쨌쩌쩍쩐쩔쩡쪼쪽쫓쬐쭈쭉쭝쭤쯔쯤찌찍찐찔찢차착찬찮찰참찹찻창찾채책챈챌챔챗챙챠처척천철첨첩첫청체첸첼쳇쳐쳤초촉촌촐촘촛총촨촬최쵸추축춘출춤춥춧충춰췄췌취츄츠측츨츰츳층치칙친칠침칩칭카칸칼캄캅캇캉캐캔캘캠캡캣캥캬커컨컫컬컴컵컷컸컹케켁켄켈켐켓켜켰코콕콘콜콤콥콧콩콰콴콸쾌쾨쾰쿄쿠쿡쿤쿨쿰쿼퀀퀄퀘퀴퀸퀼큐큘크큰클큼키킥킨킬킴킵킷킹타탁탄탈탐탑탓탔탕태택탠탤탬탭탱탸터턱턴털텀텁텃텅테텍텐텔템텝텟텡톈토톡톤톨톰톱통퇴투툭툰툴툼퉁튀튕튜튠튬트특튼튿틀틈티틱틴틸팀팁팅파팍팎판팔팜팝팟팡팥패팩팬팰팹팻팽퍼퍽펀펄펌펑페펙펜펠펨펩펫펴편펼폄폈평폐포폭폰폴폼퐁푀표푸푹푼풀품풋풍퓌퓨퓰프픈플픔피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햇했행햐향허헉헌헐험헛헝헤헥헨헬헴헵혀혁현혈혐협혔형혜호혹혼홀홈홉홋홍화확환활황홰횃회획횟횡효후훅훈훌훑훔훗훙훤훨훼휘휜휠휩휴흉흐흑흔흘흙흠흡흥흩희흰히힉힌힐힘힙', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=256,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if not opt.exp_name:
        opt.exp_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.exp_name += f'-Seed{opt.manualSeed}'
        # print(opt.exp_name)

    os.makedirs(f'./saved_models/{opt.exp_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
   #  torch.load(map_location=torch.device('cpu'))
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu
        opt.batch_size = opt.batch_size * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
