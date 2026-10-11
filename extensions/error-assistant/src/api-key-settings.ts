import { ServerConnection } from '@jupyterlab/services';
import type { TranslationBundle } from '@jupyterlab/translation';
import { Widget } from '@lumino/widgets';

import { requestAPI } from './request';

interface IAPIKeyStatus {
  configured: boolean;
  source: 'desktop' | 'environment' | null;
}

export function createAPIKeySettingsWidget(
  settings: ServerConnection.ISettings,
  trans: TranslationBundle
): Widget {
  const widget = new Widget();
  widget.id = 'mm-api-key-settings';
  widget.title.label = trans.__('API Key');
  widget.title.caption = trans.__(
    'Manage Gemini credentials stored on this computer'
  );
  widget.addClass('mm-api-key-settings-widget');

  const root = document.createElement('section');
  root.className = 'mm-key-settings';

  const header = document.createElement('header');
  header.className = 'mm-key-settings-header';
  const title = document.createElement('h2');
  title.textContent = 'API Key';
  const subtitle = document.createElement('p');
  subtitle.textContent =
    "Gemini credentials stay in your computer's credential store.";
  header.append(title, subtitle);

  const status = document.createElement('div');
  status.className = 'mm-key-settings-status';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  status.textContent = 'Checking saved credentials...';

  const setStatus = (
    message: string,
    state: 'info' | 'success' | 'warning' | 'error' = 'info'
  ): void => {
    status.textContent = message;
    status.dataset.state = state;
  };

  const form = document.createElement('form');
  form.className = 'mm-key-settings-form';
  const label = document.createElement('label');
  label.htmlFor = 'mm-gemini-api-key';
  label.textContent = 'Google Gemini API key';
  const input = document.createElement('input');
  input.id = 'mm-gemini-api-key';
  input.type = 'password';
  input.name = 'api_key';
  input.autocomplete = 'new-password';
  input.maxLength = 512;
  input.placeholder = 'Paste your Gemini API key';
  input.required = true;
  input.setAttribute('aria-describedby', 'mm-key-settings-note');

  const actions = document.createElement('div');
  actions.className = 'mm-key-settings-actions';
  const saveButton = document.createElement('button');
  saveButton.type = 'submit';
  saveButton.textContent = 'Save key';
  const removeButton = document.createElement('button');
  removeButton.type = 'button';
  removeButton.textContent = 'Remove key';
  removeButton.disabled = true;
  actions.append(saveButton, removeButton);

  const note = document.createElement('p');
  note.id = 'mm-key-settings-note';
  note.className = 'mm-key-settings-note';
  note.textContent =
    'The key is never shown again after saving and is not written to a notebook or browser storage. It is stored on the machine running Jupyter Server, so only save it from a server you trust. Gemini requests send the selected code and error context to Google. Ollama stays local.';
  form.append(label, input, actions, note);

  root.append(header, status, form);
  widget.node.appendChild(root);

  const setBusy = (busy: boolean): void => {
    input.disabled = busy;
    saveButton.disabled = busy;
    removeButton.disabled =
      busy || removeButton.dataset.source !== 'desktop';
  };

  const refreshStatus = async (): Promise<void> => {
    try {
      const result = await requestAPI<IAPIKeyStatus>(
        'api/settings/gemini-key',
        settings
      );
      removeButton.dataset.source = result.source || '';
      removeButton.disabled = result.source !== 'desktop';
      input.disabled = false;
      saveButton.disabled = false;
      if (result.source === 'desktop') {
        setStatus(
          'Your personal Gemini key is saved in this computer’s credential store.',
          'success'
        );
      } else if (result.source === 'environment') {
        setStatus(
          'A server Gemini key is configured. Saving here will use your personal key instead.',
          'info'
        );
      } else {
        setStatus('No Gemini key saved on this computer.', 'warning');
      }
    } catch {
      setStatus(
        'Could not read the system credential store. Check the Jupyter server log.',
        'error'
      );
      input.disabled = true;
      saveButton.disabled = true;
      removeButton.disabled = true;
    }
  };

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const apiKey = input.value.trim();
    if (!apiKey) {
      setStatus('Enter a Gemini API key before saving.', 'warning');
      input.focus();
      return;
    }

    setBusy(true);
    setStatus('Saving to the system credential store...');
    try {
      await requestAPI<IAPIKeyStatus>('api/settings/gemini-key', settings, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: apiKey })
      });
      input.value = '';
      setStatus(
        "Gemini key saved in this computer's credential store.",
        'success'
      );
      window.dispatchEvent(new Event('modelmind:gemini-key-updated'));
      await refreshStatus();
    } catch {
      setStatus(
        'Could not save the key. Check that the system credential store is available.',
        'error'
      );
    } finally {
      setBusy(false);
    }
  });

  removeButton.addEventListener('click', async () => {
    setBusy(true);
    setStatus('Removing saved Gemini key...');
    try {
      await requestAPI<IAPIKeyStatus>('api/settings/gemini-key', settings, {
        method: 'DELETE'
      });
      input.value = '';
      setStatus('Saved Gemini key removed.', 'success');
      window.dispatchEvent(new Event('modelmind:gemini-key-updated'));
      await refreshStatus();
    } catch {
      setStatus(
        'Could not remove the key. Check that the system credential store is available.',
        'error'
      );
    } finally {
      setBusy(false);
    }
  });

  if (!document.getElementById('mm-key-settings-styles')) {
    const style = document.createElement('style');
    style.id = 'mm-key-settings-styles';
    style.textContent = `
      .mm-key-settings {
        box-sizing: border-box;
        width: 100%; min-width: 0; height: 100%;
        overflow-y: auto; padding: 18px 14px;
        color: #e2e8f0; background: #0f172a;
        font-family: 'Segoe UI', system-ui, sans-serif;
      }
      .mm-key-settings-header {
        margin: -18px -14px 16px; padding: 18px 14px 16px;
        background: linear-gradient(135deg, #1e1b4b, #0f172a);
        border-bottom: 1px solid rgba(167,139,250,.2);
      }
      .mm-key-settings-header h2 { margin: 0; color: #f1f5f9; font-size: 16px; }
      .mm-key-settings-header p, .mm-key-settings-note {
        color: #aab5c8; font-size: 11px; line-height: 1.6;
      }
      .mm-key-settings-header p { margin: 6px 0 0; }
      .mm-key-settings-status {
        box-sizing: border-box; width: 100%; padding: 10px;
        border: 1px solid rgba(167,139,250,.3);
        border-radius: 10px; color: #c4b5fd; background: rgba(30,27,75,.4);
        font-size: 12px; line-height: 1.5;
      }
      .mm-key-settings-status[data-state="success"] {
        border-color: rgba(167,139,250,.38); color: #c4b5fd;
        background: rgba(30,27,75,.48);
      }
      .mm-key-settings-status[data-state="warning"] {
        border-color: rgba(245,158,11,.35); color: #fcd34d;
        background: rgba(120,80,10,.16);
      }
      .mm-key-settings-status[data-state="error"] {
        border-color: rgba(248,113,113,.38); color: #fca5a5;
        background: rgba(127,29,29,.2);
      }
      .mm-key-settings-form { display: flex; flex-direction: column; gap: 9px; margin-top: 18px; }
      .mm-key-settings-form label { color: #dbe4f0; font-size: 11px; font-weight: 600; }
      .mm-key-settings-form input {
        box-sizing: border-box; width: 100%; min-width: 0; padding: 10px;
        border: 1px solid #334155; border-radius: 8px;
        outline: none; color: #f8fafc; background: #1e293b;
        font: inherit; font-size: 13px;
      }
      .mm-key-settings-form input:focus { border-color: #a78bfa; }
      .mm-key-settings-form input::placeholder { color: #aab5c8; opacity: 1; }
      .mm-key-settings-actions { display: flex; gap: 8px; }
      .mm-key-settings-actions button {
        box-sizing: border-box; min-width: 0; min-height: 38px; flex: 1;
        padding: 8px; border: 1px solid #5546a5; border-radius: 8px;
        color: #fff; background: #43358a; font: inherit; font-size: 12px;
        font-weight: 600; cursor: pointer;
      }
      .mm-key-settings-actions button:last-child {
        border-color: #475569; color: #cbd5e1; background: #1e293b;
      }
      .mm-key-settings-actions button:hover:not(:disabled) { filter: brightness(1.12); }
      .mm-key-settings-actions button:disabled { cursor: not-allowed; opacity: .5; }
      .mm-key-settings-note { margin: 2px 0 0; }
    `;
    document.head.appendChild(style);
  }

  void refreshStatus();
  return widget;
}
