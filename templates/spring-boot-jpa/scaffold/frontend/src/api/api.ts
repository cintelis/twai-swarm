import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react'
import type { RootState } from '../store'

interface LoginRequest {
  username: string
  password: string
}

interface LoginResponse {
  token: string
}

interface HelloResponse {
  status: string
  message: string
}

interface MeResponse {
  username: string
}

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({
    baseUrl: '/api',
    prepareHeaders: (headers, { getState }) => {
      const token = (getState() as RootState).auth.token
      if (token) headers.set('authorization', `Bearer ${token}`)
      return headers
    },
  }),
  endpoints: (builder) => ({
    login: builder.mutation<LoginResponse, LoginRequest>({
      query: (body) => ({ url: '/auth/login', method: 'POST', body }),
    }),
    hello: builder.query<HelloResponse, void>({
      query: () => '/hello',
    }),
    me: builder.query<MeResponse, void>({
      query: () => '/hello/me',
    }),
  }),
})

export const { useLoginMutation, useHelloQuery, useMeQuery } = api
