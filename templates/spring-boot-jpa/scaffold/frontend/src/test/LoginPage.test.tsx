import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Provider } from 'react-redux'
import { BrowserRouter } from 'react-router-dom'
import LoginPage from '../pages/LoginPage'
import store from '../store'
import * as apiModule from '../api/api'

vi.mock('../api/api', async () => {
  const actual = await vi.importActual('../api/api')
  return {
    ...actual,
    useLoginMutation: vi.fn(),
  }
})

const renderWithProviders = (component: React.ReactElement) => {
  return render(
    <Provider store={store}>
      <BrowserRouter>
        {component}
      </BrowserRouter>
    </Provider>
  )
}

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders login form with username, password inputs and login button', () => {
    vi.mocked(apiModule.useLoginMutation).mockReturnValue([
      vi.fn(),
      { isLoading: false },
    ] as any)

    renderWithProviders(<LoginPage />)

    expect(screen.getByTestId('username-input')).toBeInTheDocument()
    expect(screen.getByTestId('password-input')).toBeInTheDocument()
    expect(screen.getByTestId('login-button')).toBeInTheDocument()
  })

  it('shows error message on failed login', async () => {
    const user = userEvent.setup()
    const loginMock = vi.fn().mockReturnValue({
      unwrap: () => Promise.reject(new Error('Invalid credentials')),
    })

    vi.mocked(apiModule.useLoginMutation).mockReturnValue([
      loginMock,
      { isLoading: false },
    ] as any)

    renderWithProviders(<LoginPage />)

    const usernameInput = screen.getByTestId('username-input')
    const passwordInput = screen.getByTestId('password-input')
    const loginButton = screen.getByTestId('login-button')

    await user.type(usernameInput, 'testuser')
    await user.type(passwordInput, 'wrongpassword')
    await user.click(loginButton)

    await waitFor(() => {
      expect(screen.getByTestId('error-message')).toBeInTheDocument()
      expect(screen.getByTestId('error-message')).toHaveTextContent('Invalid credentials')
    })
  })

  it('successful login dispatches token and navigates', async () => {
    const user = userEvent.setup()
    const loginMock = vi.fn().mockReturnValue({
      unwrap: () => Promise.resolve({ token: 'test-token-123' }),
    })

    vi.mocked(apiModule.useLoginMutation).mockReturnValue([
      loginMock,
      { isLoading: false },
    ] as any)

    renderWithProviders(<LoginPage />)

    const usernameInput = screen.getByTestId('username-input')
    const passwordInput = screen.getByTestId('password-input')
    const loginButton = screen.getByTestId('login-button')

    await user.type(usernameInput, 'testuser')
    await user.type(passwordInput, 'correctpassword')
    await user.click(loginButton)

    await waitFor(() => {
      expect(loginMock).toHaveBeenCalledWith({
        username: 'testuser',
        password: 'correctpassword',
      })
    })
  })
})
