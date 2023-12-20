# # Fast inference with vLLM (Swallow 13B)
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# `vLLM` also supports a use case as a FastAPI server which we will explore in a future guide. This example
# walks through setting up an environment that works with `vLLM ` for basic inference.
#
# To run
# [any of the other supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html),
# simply replace the model name in the download step. You may also need to enable `trust_remote_code` for MPT models (see comment below)..
#
# ## Setup
#
# First we import the components we need from `modal`.

import os

from modal import Image, Stub, method

MODEL_DIR = "/model"
BASE_MODEL = "TheBloke/Swallow-13B-Instruct-AWQ"


# ## Define a container image
#
# We want to create a Modal image which has the model weights pre-saved to a directory. The benefit of this
# is that the container no longer has to re-download the model from Huggingface - instead, it will take
# advantage of Modal's internal filesystem for faster cold starts.
#
# ### Download the weights
# Make sure you have created a [HuggingFace access token](https://huggingface.co/settings/tokens).
#
# We can download the model to a particular directory using the HuggingFace utility function `snapshot_download`.
#
# Tip: avoid using global variables in this function. Changes to code outside this function will not be detected and the download step will not re-run.
def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
    )
    move_cache()


# ### Image definition
# We’ll start from a recommended Dockerhub image and install `vLLM`.
# Then we’ll use run_function to run the function defined above to ensure the weights of
# the model are saved within the container image.
image = (
    Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
    )
    .pip_install("vllm==0.2.6", "huggingface_hub==0.19.4", "hf-transfer==0.1.4")
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        timeout=60 * 20,
    )
)

stub = Stub("vllm-inference-swallow-13B", image=image)


# ## The model class
#
# The inference function is best represented with Modal's [class syntax](/docs/guide/lifecycle-functions) and the `__enter__` method.
# This enables us to load the model into memory just once every time a container starts up, and keep it cached
# on the GPU for each subsequent invocation of the function.
#
# The `vLLM` library allows the code to remain quite clean.
@stub.cls(gpu="A100")
class Model:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR, quantization="awq", dtype="auto")
        self.template = """以下に、あるタスクを説明する指示があります。リクエストを適切に完了するための回答を記述してください。

### 指示:
{prompt}

### 応答:
"""

    @method()
    def generate(self, user_questions):
        from vllm import SamplingParams

        prompts = [
            self.template.format(prompt=q) for q in user_questions
        ]

        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=800,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        num_tokens = 0
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(output.prompt, output.outputs[0].text, "\n\n", sep="")
        print(f"Generated {num_tokens} tokens")


# ## Run the model
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        # プログラミング関連の質問
        "Fibonacci数を計算するPython関数を実装してください。",
        "二進数指数関数を行うRust関数を書いてください。",
        "C言語でどのようにメモリを割り当てますか？",
        "JavascriptとPythonの違いは何ですか？",
        "Postgresで無効なインデックスをどのように見つけますか？",
        "PythonでLRU（最近最も使われていない）キャッシュをどのように実装しますか？",
        "マルチスレッドアプリケーションで競合状態を検出し、防止するためにどのようなアプローチを取りますか？",
        "機械学習における決定木アルゴリズムの動作を説明してください。",
        "どのようにして単純なキー値ストアデータベースを一から設計しますか？",
        "並行プログラミングにおけるデッドロック状況をどのように処理しますか？",
        "A*検索アルゴリズムの背後にあるロジックと、それがどこで使用されるかは何ですか？",
        "効率的なオートコンプリートシステムをどのように設計しますか？",
        "Webアプリケーションで安全なセッション管理システムを設計するためのアプローチは何ですか？",
        "ハッシュテーブルで衝突をどのように処理しますか？",
        "分散システム用の負荷分散装置をどのように実装しますか？",
        # 文学
        "キツネとブドウに関する寓話は何ですか？",
        "2083年にオーストラリアの砂漠への旅行について、ジェームズ・ジョイス風の物語を書いてください。美しい砂漠でロボットを見るためです。",
        "ハリーは誰を風船に変えますか？",
        "人類史上で最も重要な出来事を目撃することに決めた時間旅行する歴史家についての物語を書いてください。",
        "秘密エージェントでありながらフルタイムの親でもある人の一日を描写してください。",
        "動物とコミュニケーションをとることができる探偵についての物語を作成してください。",
        "雲の上に浮かぶ都市に住むことについて最も珍しいことは何ですか？",
        "夢が共有される世界で、平和な夢に悪夢が侵入したらどうなりますか？",
        "平行宇宙に通じる地図を見つけた友人たちの一生に一度の冒険を描写してください。",
        "自分の音楽に魔法の力があることを発見したミュージシャンについての物語を語ってください。",
        "人々が逆に歳を取る世界で、5歳の男性の人生を描写してください。",
        "毎晩絵が生き生きとする画家についての物語を作成してください。",
        "詩人の詩が未来の出来事を予言し始めたらどうなりますか？",
        "本が話すことができる世界を想像してください。図書館員はどのように対処しますか？",
        "植物によって住まわれた惑星を発見した宇宙飛行士についての物語を語ってください。",
        "これまでで最も洗練された郵便サービスを通じて旅する手紙の旅を描写してください。",
        "食べる人の過去からの思い出を呼び起こすことができるシェフについての物語を書いてください。",
        # 歴史
        "ローマ帝国の衰退に大きく貢献した要因は何でしたか？",
        "印刷プレスの発明がヨーロッパ社会に革命をもたらした方法は？",
        "量的緩和の影響は何ですか？",
        "古代世界における経済思想にギリシャ哲学者はどのような影響を与えましたか？",
        "ソビエト連邦の崩壊につながった経済的・哲学的な要因は何でしたか？",
        "20世紀の非植民地化は地政学的な地図をどのように変えましたか？",
        "クメール帝国は東南アジアの歴史と文化にどのような影響を与えましたか？",
        # 思慮深さ
        "技術の歩、環境の変化、社会の変化を考慮して、未来の都市を描写してください。",
        "水が最も価値のある商品となるディストピア的な未来では、社会はどのように機能しますか？",
        "科学者が不老不死を発見した場合、社会、経済、環境にどのような影響を与える可能性がありますか？",
        "高度な宇宙文明との接触がもたらす可能性のある影響は何ですか？",
        # 数学
        "9と8の積は何ですか？",
        "列車が2時間で120キロメートルを走行した場合、その平均速度は何ですか？",
        "このステップを一歩一歩考えてください。数列a_nがa_1 = 3、a_2 = 5、そしてn > 2に対してa_n = a_(n-1) + a_(n-2)と定義される場合、a_6を求めてください。",
        "このステップを一歩一歩考えてください。初項3、最終項35、総項数11の算術級数の合計を計算してください。",
        "このステップを一歩一歩考えてください。点(1,2)、(3,-4)、(-2,5)における三角形の面積はいくつですか？",
        "このステップを一歩一歩考えてください。次の線形方程式系を解いてください：3x + 2y = 14, 5x - y = 15。",
        # 事実
        "エンペラー・ノートンIは誰であり、サンフランシスコの歴史における彼の重要性は何ですか？",
        "ヴォイニッチ手稿とは何であり、なぜ何世紀にもわたって学者を困惑させてきたのですか？",
        "プロジェクトA119とは何であり、その目的は何でしたか？",
        "ディアトロフ峠事件とは何であり、なぜそれは謎のままですか？",
        "1930年代のオーストラリアで起こった「エミュー戦争」とは何ですか？",
        "ヘリベルト・イリッヒによって提案された「ファントム・タイム・ハイポセシス」とは何ですか？",
        "12世紀のイングランド伝説による「ウールピットの緑の子供たち」とは誰ですか？",
        "天文学における「ゾンビ星」とは何ですか？",
        "中世キリスト教伝統の「犬頭聖者」と「ライオン顔の聖者」とは誰ですか？",
        "岸辺に打ち上げられた未確認の有機質塊である「グロブスターズ」の物語とは何ですか？",
    ]
    model.generate.remote(questions)
