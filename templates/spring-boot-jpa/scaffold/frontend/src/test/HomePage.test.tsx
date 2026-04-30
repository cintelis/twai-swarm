import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { configureStore } from '@reduxjs/toolkit'
import authReducer from '../api/authSlice'
import { api } from '../api/api'
import HomePage from '../pages/HomePage'

vi.mock('../api/api', async () => {
  const actual = await vi.importActual<typeof import('../api/api')>('../api/api')
  return {
    ...actual,
    useHelloQuery: vi.fn(() => ({ data: undefined, isLoading: true, error: undefined })),
    useMeQuery: vi.fn(() => ({ data: undefined, isLoading: true, error: undefined })),
  }
})

function renderWithProviders() {
  const store = configureStore({
    reducer: {
      auth: authReducer,
      [api.reducerPath]: api.reducer,
    },
    middleware: (getDefaultMiddleware) =>
      getDefaultMiddleware().concat(api.middleware),
  })
  return render(
    <Provider store={store}>
      <MemoryRouter>
        <HomePage />
      </MemoryRouter>
    </Provider>,
  )
}

describe('HomePage', () => {
  it('renders both endpoint sections and a logout button', () => {
    renderWithProviders()
    expect(screen.getByText('Home')).toBeInTheDocument()
    expect(screen.getByTestId('hello-section')).toBeInTheDocument()
    expect(screen.getByTestId('me-section')).toBeInTheDocument()
    expect(screen.getByTestId('logout-button')).toBeInTheDocument()
  })
})
