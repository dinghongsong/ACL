#!/usr/bin/env python3
import boto3
import json
import sys

def test_bedrock():
    try:
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name='us-east-1'
        )
        
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
                "messages": [
                    {"role": "user", "content": "你好，请用一句话介绍自己。"}
                ]
            })
        )
        
        result = json.loads(response['body'].read())
        print("✓ 成功连接 AWS Bedrock!")
        print(f"✓ 模型响应: {result.get('content', [{}])[0].get('text', result)}")
        return True
        
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("\n请检查:")
        print("1. AWS 凭证配置: aws configure")
        print("2. 区域是否支持 Bedrock")
        print("3. 是否有 Bedrock 访问权限")
        return False

if __name__ == "__main__":
    success = test_bedrock()
    sys.exit(0 if success else 1)
