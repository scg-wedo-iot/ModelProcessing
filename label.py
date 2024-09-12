
def checkIsAnyServiceCorrect(payload, listService, preword='is_correct_'):
    header_picture = f'{preword}picture'
    isAnyServiceCorrect = False
    service_correct = 'human'

    if not payload[header_picture] is None:
        for i in range(len(listService)):
            service_check = listService[i]
            header_service = f'{preword}{service_check}'
            service_status = payload[header_service]
            if (not service_status is None) and service_status:
                isAnyServiceCorrect = True
                service_correct = service_check
                break

    return isAnyServiceCorrect, service_correct