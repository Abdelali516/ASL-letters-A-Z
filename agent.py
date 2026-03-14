import ollama

save_conversation=[]

def function(model_type,user_input):

    save_conversation.append({
        'role':'user','content':user_input
    })

    model_reponse=ollama.chat(
        messages=save_conversation,
        model=model_type,
        stream=True
    )

    full_reponse=""
    for chunk in model_reponse:
        if chunk.message.content:
            response=chunk.message.content
            full_reponse+=response
            print(response,end="",flush=True)
    print()
    save_conversation.append({
        'role':'assistant','content':full_reponse
    })

    return full_reponse
