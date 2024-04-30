from django.shortcuts import render
from django.http import JsonResponse
from agent.agent import handle_userinput


def ask_openai(message):
    response = handle_userinput(message)
    print(response)
    return response['output']

    

def chatbot(request):
    if request.method == 'POST':
        message = request.POST.get('message')
        response = ask_openai(message)
        return JsonResponse({'message': message, 'response': response})
        
    return render(request, 'chatbot.html')
