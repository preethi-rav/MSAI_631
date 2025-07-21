#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


""" Bot Configuration """


class DefaultConfig:
    """ Bot Configuration """

    PORT = 3980
    APP_ID = os.environ.get("MicrosoftAppId", "")
    APP_PASSWORD = os.environ.get("MicrosoftAppPassword", "")
    APP_TYPE = os.environ.get("MicrosoftAppType", "MultiTenant")
    APP_TENANTID = os.environ.get("MicrosoftAppTenantId", "")

    endpoint = os.environ.get("MicrosoftAIServiceEndpoint")
    print("Endpoint:", os.environ.get("MicrosoftAIServiceEndpoint"))
    print("API Key:", os.environ.get("MicrosoftAPIKey"))

    key = os.environ.get("MicrosoftAPIKey")
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=AzureKeyCredential(key))
