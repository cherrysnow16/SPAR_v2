#!/bin/bash
# Qwen3.5-2B WebShop: SPAR v2, then the GraphGPO baseline, one after the other on both GPUs
cd /workspace/SPAR
bash logs/launch_qwen35_webshop.sh > logs/spar_v2_qwen3.5_2b_webshop_textrich.log 2>&1
ray stop --force >/dev/null 2>&1 || true
sleep 30
bash logs/launch_graphgpo_qwen35_webshop.sh > logs/graphgpo_qwen3.5_2b_webshop_textrich.log 2>&1
